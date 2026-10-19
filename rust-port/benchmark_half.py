#!/usr/bin/env python3
# Copyright 2026 rt-rs contributors.
# SPDX-License-Identifier: Apache-2.0
"""Profile full native FP16 conversion+greedy against safe fused Rust selection."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
from datetime import datetime, timezone
from verify import PORT, TARGET, run

SOURCES = ['src/half.rs', 'src/greedy.rs', 'src/bin/half_trace.rs',
           'tests/half_reference.cc', 'tests/half_fused_control.h', 'verify_half.py', 'benchmark_half.py',
           'Cargo.toml', 'Cargo.lock']
BRIDGE_SOURCES = ['native/greedy_rust.h', 'native/greedy_rust.cc', 'native/stop_ffi.h',
                  'ffi/src/lib.rs', 'ffi/Cargo.toml', 'ffi/Cargo.lock']
CANDIDATES = {'cpp_fused_control': ('half-reference-fused-control', 'staged'),
              'cpp_hostcopy_fused_control': ('half-reference-fused-control', 'staged-copy'),
              'cpp_staged': ('half-reference-native', 'staged'),
              'cpp_hostcopy_staged': ('half-reference-native', 'staged-copy'),
              'rust_staged': ('release/half-trace', 'staged'),
              'rust_fused': ('release/half-trace', 'fused'),
              'rust_hostcopy_fused': ('release/half-trace', 'fused-copy'),
              'rust_fused_previous': ('release/half-trace-branchy', 'fused')}

BRIDGE_CANDIDATES = {
    'native_rust_fused': ('half-reference-ffi', 'staged'),
    'native_rust_hostcopy_fused': ('half-reference-ffi', 'staged-copy')}


def hashes(bridge=False):
    return {p: hashlib.sha256((PORT / p).read_bytes()).hexdigest()
            for p in SOURCES + (BRIDGE_SOURCES if bridge else [])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['build', 'run', 'all'], default='all')
    parser.add_argument('--pattern', choices=['random', 'ascending', 'periodic'], default='random')
    parser.add_argument('--ffi', action='store_true',
                        default=os.environ.get('RT_RS_HALF_FFI') == '1',
                        help='Include standalone native-to-Rust binary16 boundary')
    args = parser.parse_args()
    candidates = dict(CANDIDATES)
    if args.ffi:
        candidates.update(BRIDGE_CANDIDATES)
    build_path = TARGET / 'half-bench-build.json'
    if args.phase in ('build', 'all'):
        env = dict(os.environ, RT_RS_HALF_FFI='1' if args.ffi else '0')
        run(['python3', PORT / 'verify_half.py'], env=env)
        report = json.loads((PORT / 'half-results.json').read_text())
        report['source_sha256'] = hashes(args.ffi)
        previous_tree = 'd8ffb7957f35a6ac7ecf0fbb88e58450d314044e'
        previous_root = TARGET / 'half-previous-source'
        paths = run(['git', 'ls-tree', '-r', '--name-only', previous_tree, '--',
                     'rust-port/Cargo.toml', 'rust-port/Cargo.lock', 'rust-port/src']).splitlines()
        for path in paths:
            destination = previous_root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(run(['git', 'show', f'{previous_tree}:{path}']))
        previous_build = TARGET / 'half-previous-build'
        run(['cargo', 'build', '--release', '--manifest-path', previous_root / 'rust-port/Cargo.toml',
             '--target-dir', previous_build, '--offline', '--locked'])
        shutil.copy2(previous_build / 'release/half-trace', TARGET / 'release/half-trace-branchy')
        report['previous_fused_tree'] = previous_tree
        report['previous_rebuilt_source_sha256'] = {
            p: hashlib.sha256((previous_root / p).read_bytes()).hexdigest() for p in paths}
        report['previous_fused_source_sha256'] = json.loads((PORT / 'benchmarks/cloud-half-branchy-random.json').read_text())['source_sha256']
        report['binary_sha256'].update({p: hashlib.sha256((TARGET / p).read_bytes()).hexdigest()
                                   for p, _ in set(candidates.values())})
        build_path.write_text(json.dumps(report, indent=2) + '\n')
    if args.phase == 'build':
        print('Exhaustive conversion and reduction parity checked; benchmark ready.')
        return
    report = json.loads(build_path.read_text())
    assert report['native_boundary_checked'] == args.ffi, 'Bridge mode changed; rebuild'
    assert report['source_sha256'] == hashes(args.ffi), 'Sources changed; rebuild'
    report['candidates'] = candidates
    for p, expected in report['binary_sha256'].items():
        assert hashlib.sha256((TARGET / p).read_bytes()).hexdigest() == expected
    report.update({'measurement_utc': datetime.now(timezone.utc).isoformat(),
                   'platform': platform.platform(), 'input_pattern': args.pattern,
                   'scope': 'Native conversion+selection versus safe fused half reduction; hostcopy means host allocation+copy, never GPU/device readback. Optional standalone native-to-Rust boundary includes flat output allocation and nested result reconstruction. No full-engine/model integration.',
                   'timing_status': 'Provisional shared-cloud-host microbenchmarks; known project workers idle, no affinity/thermal/frequency control. GCC versus Rust LLVM.',
                   'samples': []})
    for vocab in (32768, 131072, 262144):
        for batch, sequence in ((1, 1), (4, 1), (1, 8), (4, 8)):
            rounds = max(2, 2_000_000 // (vocab * batch * sequence))
            samples = {k: [] for k in candidates}
            checksums = set()
            for repeat in range(7):
                order = list(candidates)
                order = order[repeat % len(order):] + order[:repeat % len(order)]
                for name in order:
                    binary, mode = candidates[name]
                    ns, checksum = map(int, run([TARGET / binary, mode, batch, sequence, vocab, rounds, args.pattern]).split())
                    checksums.add(checksum)
                    if repeat >= 2: samples[name].append(ns / (rounds * batch * sequence))
            assert len(checksums) == 1, 'Half selection indices differ'
            row = {'vocab': vocab, 'batch': batch, 'sequence': sequence, 'rounds': rounds,
                   'checksum': checksums.pop(), 'ns_per_position_samples': samples,
                   'ns_per_position_median': {k: statistics.median(v) for k, v in samples.items()},
                   'float32_staging_bytes': vocab * batch * sequence * 4,
                   'optional_half_hostcopy_bytes': vocab * batch * sequence * 2}
            report['samples'].append(row)
            print(vocab, batch, sequence, row['ns_per_position_median'], flush=True)
    (PORT / 'half-benchmark-results.json').write_text(json.dumps(report, indent=2) + '\n')

if __name__ == '__main__': main()
