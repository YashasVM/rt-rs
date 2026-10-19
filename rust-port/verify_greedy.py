#!/usr/bin/env python3
# Copyright 2026 rt-rs contributors.
# SPDX-License-Identifier: Apache-2.0
"""Differential-test the Rust greedy port against unchanged native sampling."""
import hashlib
import json
import os
import platform
import random
import shlex
import struct

from verify import BASELINE, CURRENT_MAIN, PORT, ROOT, TARGET, run


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def main():
    current_main = os.environ.get("RT_RS_CURRENT_MAIN") == "1"
    for path in ['runtime/components/sampling_cpu_util.cc',
                 'runtime/components/sampling_cpu_util.h']:
        if (ROOT / path).read_text() != run(['git', 'show', f'{BASELINE}:{path}']):
            raise SystemExit(f'Reference differs from baseline: {path}')
        if current_main and (ROOT / path).read_text() != run(['git', 'show', f'{CURRENT_MAIN}:{path}']):
            raise SystemExit(f'Current main sampling source has changed: {path}')
    run(['cargo', 'test', '--manifest-path', PORT / 'Cargo.toml', '--locked', '--offline'])
    run(['cargo', 'build', '--release', '--manifest-path', PORT / 'Cargo.toml', '--locked', '--offline'])
    cpp = os.environ.get('CXX', 'c++')
    flags = shlex.split(run(['pkg-config', '--cflags', '--libs', '--static',
                            'absl_status', 'absl_statusor', 'absl_strings', 'absl_str_format']))
    reference = TARGET / 'greedy-reference'
    extra_flags = shlex.split(os.environ.get('CXXFLAGS', ''))
    run([cpp, '-std=c++20', '-O3', '-DNDEBUG', *extra_flags, '-I', ROOT,
         PORT / 'tests/greedy_reference.cc',
         ROOT / 'runtime/components/sampling_cpu_util.cc', '-o', reference, *flags])
    cases = []

    def add(values, batch=1, sequence=1):
        cases.append(f'{batch} {sequence} {len(values)} ' +
                     ' '.join(f'{value:08x}' for value in values) + '\n')

    special = [bits(0.0), bits(-0.0), bits(float('inf')), bits(float('-inf')),
               0x7fc00000, 0xffc00001, 0x7f800001, 0x00000001, 0x80000001]
    sizes = [0, 1, 2, 7, 8, 9, 127, 128, 129, 255, 256, 257, 383, 384, 385, 1024]
    for size in sizes:
        for value in special + [bits(1), bits(-1)]:
            add([value] * size)
        add([bits(float(i)) for i in range(size)])
        add([bits(float(-i)) for i in range(size)])
        for position in range(size):
            values = [bits(-1)] * size
            values[position] = bits(10)
            add(values)
            # First-tie behavior at all positions, including across tile edges.
            if size:
                values[-1] = bits(10)
                add(values)
        # Every lane and tail receives a NaN; a higher value later in its lane
        # distinguishes the literal tiled C++ algorithm from mathematical argmax.
        for position in range(size):
            values = [bits(float(i % 19)) for i in range(size)]
            values[position] = 0x7fc00000
            add(values)
    rng = random.Random(18001)
    for _ in range(1200):
        batch, sequence = rng.randint(1, 4), rng.randint(1, 4)
        size = rng.choice(sizes)
        values = ([rng.getrandbits(32) for _ in range(batch * sequence * size)]
                  if rng.random() < .5 else
                  [bits(float(rng.randint(-10, 10))) for _ in range(batch * sequence * size)])
        add(values, batch, sequence)
    for batch, sequence in [(1, 2), (2, 1), (2, 3), (4, 4)]:
        add([], batch, sequence)
        add([bits(1)], batch, sequence)
    inputs = ''.join(cases)
    expected = run([reference], input=inputs)
    actual = run([TARGET / 'release/greedy-trace'], input=inputs)
    if expected != actual:
        (TARGET / 'failed-greedy-traces.txt').write_text(inputs)
        for index, (left, right) in enumerate(zip(expected.splitlines(), actual.splitlines())):
            if left != right:
                (TARGET / 'failed-greedy-trace.txt').write_text(cases[index])
                raise AssertionError(f'Case {index}: C++={left}, Rust={right}')
        raise AssertionError('Output line count mismatch')
    report = {'baseline': CURRENT_MAIN if current_main else BASELINE,
              'compatibility_baseline': BASELINE, 'cases': len(cases),
              'input_sha256': hashlib.sha256(inputs.encode()).hexdigest(),
              'platform': platform.platform(),
              'rustc': run(['rustc', '--version']).strip(),
              'cxx': run([cpp, '--version']).splitlines()[0],
              'extra_cxxflags': extra_flags,
              'abseil': run(['pkg-config', '--modversion', 'absl_status']).strip(),
              'scope': 'TopKTokenIds k=1; positive shapes within C++ integer range; no runtime integration'}
    (PORT / 'greedy-results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
