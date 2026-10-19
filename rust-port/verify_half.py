#!/usr/bin/env python3
# Copyright 2026 rt-rs contributors.
# SPDX-License-Identifier: Apache-2.0
"""Exhaustive native half conversion and fused-reduction differential checks."""
import hashlib
import json
import os
import platform
import random
import shlex
from verify import BASELINE, CURRENT_MAIN, LITERT_BASELINE, LITERT, PORT, ROOT, TARGET, run


def conversion_body(source):
    start = source.index('void ConvertFp16ToFp32(')
    return source[start:source.index('\n}', start) + 2]


def main():
    source = 'runtime/components/top_p_cpu_sampler.cc'
    body = conversion_body(run(['git', 'show', f'{BASELINE}:{source}']))
    assert body == conversion_body(run(['git', 'show', f'{CURRENT_MAIN}:{source}']))
    assert body == conversion_body((PORT / 'tests/half_reference.cc').read_text())
    for path in ['runtime/components/sampling_cpu_util.cc', 'runtime/components/sampling_cpu_util.h']:
        assert (ROOT / path).read_text() == run(['git', 'show', f'{BASELINE}:{path}'])
        assert (ROOT / path).read_text() == run(['git', 'show', f'{CURRENT_MAIN}:{path}'])
    assert run(['git', '-C', LITERT, 'rev-parse', 'HEAD']).strip() == LITERT_BASELINE
    run(['git', '-C', LITERT, 'diff', '--exit-code', 'HEAD', '--', 'tflite/types'])
    run(['cargo', 'test', '--manifest-path', PORT / 'Cargo.toml', '--offline', '--locked'])
    run(['cargo', 'build', '--release', '--manifest-path', PORT / 'Cargo.toml', '--offline', '--locked'])
    cpp = os.environ.get('CXX', 'c++')
    flags = shlex.split(run(['pkg-config', '--cflags', '--libs', '--static',
                            'absl_status', 'absl_statusor', 'absl_strings', 'absl_str_format']))
    reference = TARGET / 'half-reference-native'
    extra = shlex.split(os.environ.get('CXXFLAGS', ''))
    run([cpp, '-std=c++20', '-O3', '-DNDEBUG', *extra, '-I', ROOT, '-I', LITERT,
         PORT / 'tests/half_reference.cc', ROOT / 'runtime/components/sampling_cpu_util.cc',
         '-o', reference, *flags])
    rust = TARGET / 'release/half-trace'
    conversion = run([reference, 'convert'])
    actual = run([rust, 'convert'])
    assert len(conversion.splitlines()) == 65536
    if conversion != actual:
        for bits, (left, right) in enumerate(zip(conversion.splitlines(), actual.splitlines())):
            if left != right: raise AssertionError(f'Half {bits:04x}: native={left} Rust={right}')
        raise AssertionError('Conversion output length differs')
    cases = []
    def add(values, batch=1, sequence=1):
        cases.append(f'{batch} {sequence} {len(values)} ' + ' '.join(f'{v:04x}' for v in values) + '\n')
    special = [0, 0x8000, 1, 0x3ff, 0x400, 0xbc00, 0x3c00, 0x7bff, 0xfbff,
               0x7c00, 0xfc00, 0x7c01, 0x7e00, 0xfc01, 0xfe00]
    sizes = [0, 1, 7, 8, 9, 127, 128, 129, 255, 256, 257, 1024]
    for size in sizes:
        for value in special: add([value] * size)
        for position in range(size):
            values = [0xbc00] * size
            values[position] = 0x7bff
            add(values)
            values[-1] = 0x7bff
            add(values)
            values = [0x3000 + (i % 99) for i in range(size)]
            values[position] = 0x7c01
            add(values)
            values[position] = 0xfc01
            add(values)
    rng = random.Random(180016)
    for _ in range(2000):
        batch, sequence = rng.randint(1, 4), rng.randint(1, 4)
        vocab = rng.choice(sizes)
        add([rng.randrange(65536) for _ in range(batch * sequence * vocab)], batch, sequence)
    all_values = list(range(65536)); add(all_values)
    all_values.reverse(); add(all_values)
    rng.shuffle(all_values); all_values[0] = 0; add(all_values)
    for batch, sequence in [(1, 2), (2, 1), (2, 3), (4, 4)]:
        add([], batch, sequence); add([0x3c00], batch, sequence)
    inputs = ''.join(cases)
    expected, actual = run([reference], input=inputs), run([rust], input=inputs)
    if expected != actual:
        for i, (left, right) in enumerate(zip(expected.splitlines(), actual.splitlines())):
            if left != right:
                (TARGET / 'failed-half-trace.txt').write_text(cases[i])
                raise AssertionError(f'Half selection case {i}: native={left} Rust={right}')
        raise AssertionError('Selection output length differs')
    native_boundary = os.environ.get('RT_RS_HALF_FFI') == '1'
    binaries = [reference, rust]
    sources = ['src/half.rs', 'src/greedy.rs', 'src/bin/half_trace.rs',
               'tests/half_reference.cc', 'verify_half.py', 'Cargo.toml', 'Cargo.lock']
    if native_boundary:
        run(['cargo', 'test', '--manifest-path', PORT / 'ffi/Cargo.toml', '--locked', '--offline'])
        run(['cargo', 'build', '--release', '--manifest-path', PORT / 'ffi/Cargo.toml',
             '--target-dir', TARGET / 'ffi', '--locked', '--offline'])
        bridge = TARGET / 'half-reference-ffi'
        archive = TARGET / 'ffi/release/librt_rs_stop_ffi.a'
        run([cpp, '-std=c++20', '-O3', '-DNDEBUG', *extra, '-DRT_RS_HALF_FFI',
             '-I', ROOT, '-I', LITERT, PORT / 'tests/half_reference.cc',
             PORT / 'native/greedy_rust.cc', archive, '-o', bridge,
             *flags, '-ldl', '-lpthread', '-lm'])
        bridge_actual = run([bridge], input=inputs)
        if bridge_actual != expected:
            for i, (left, right) in enumerate(zip(expected.splitlines(), bridge_actual.splitlines())):
                if left != right:
                    (TARGET / 'failed-half-ffi-trace.txt').write_text(cases[i])
                    raise AssertionError(f'Half bridge case {i}: native={left} bridge={right}')
            raise AssertionError('Half bridge output length differs')
        # Even this build must preserve the real upstream conversion oracle.
        assert run([bridge, 'convert']) == conversion
        binaries += [bridge, archive]
        sources += ['native/greedy_rust.h', 'native/greedy_rust.cc', 'native/stop_ffi.h',
                    'ffi/src/lib.rs', 'ffi/Cargo.toml', 'ffi/Cargo.lock']
    control = TARGET / 'half-reference-fused-control'
    run([cpp, '-std=c++20', '-O3', '-DNDEBUG', *extra, '-DRT_RS_HALF_NATIVE_FUSED',
         '-I', ROOT, '-I', LITERT, PORT / 'tests/half_reference.cc',
         '-o', control, *flags])
    assert run([control], input=inputs) == expected, 'Native fused control differs'
    assert run([control, 'convert']) == conversion
    binaries.append(control)
    sources.append('tests/half_fused_control.h')
    report = {'baseline': CURRENT_MAIN, 'compatibility_baseline': BASELINE, 'litert': LITERT_BASELINE,
              'conversion_encodings': 65536, 'conversion_sha256': hashlib.sha256(conversion.encode()).hexdigest(),
              'cases': len(cases), 'traces_sha256': hashlib.sha256(inputs.encode()).hexdigest(),
              'platform': platform.platform(), 'rustc': run(['rustc', '--version']).strip(),
              'cxx': run([cpp, '--version']).splitlines()[0], 'extra_cxxflags': extra,
              'native_fused_control_cases': len(cases),
              'native_boundary_checked': native_boundary,
              'native_boundary_cases': len(cases) if native_boundary else 0,
              'abseil': run(['pkg-config', '--modversion', 'absl_status']).strip(),
              'binary_sha256': {str(p.relative_to(TARGET)): hashlib.sha256(p.read_bytes()).hexdigest() for p in binaries},
              'source_sha256': {p: hashlib.sha256((PORT / p).read_bytes()).hexdigest() for p in sources},
              'scope': 'Safe fused binary16 greedy selection versus pinned native conversion+TopKTokenIds k1; optional standalone native-to-Rust boundary; no engine/model/device integration'}
    (PORT / 'half-results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
