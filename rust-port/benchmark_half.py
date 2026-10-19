#!/usr/bin/env python3
# Copyright 2026 rt-rs contributors.
# SPDX-License-Identifier: Apache-2.0
"""Profile full native FP16 conversion+greedy against safe fused Rust selection."""
import argparse
import hashlib
import json
import platform
import statistics
from datetime import datetime, timezone
from verify import PORT, TARGET, run

SOURCES = ['src/half.rs', 'src/greedy.rs', 'src/bin/half_trace.rs',
           'tests/half_reference.cc', 'benchmark_half.py']
CANDIDATES = {'cpp_staged': ('half-reference-native', 'staged'),
              'cpp_hostcopy_staged': ('half-reference-native', 'staged-copy'),
              'rust_staged': ('release/half-trace', 'staged'),
              'rust_fused': ('release/half-trace', 'fused'),
              'rust_hostcopy_fused': ('release/half-trace', 'fused-copy')}


def hashes():
    return {p: hashlib.sha256((PORT / p).read_bytes()).hexdigest() for p in SOURCES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['build', 'run', 'all'], default='all')
    parser.add_argument('--pattern', choices=['random', 'ascending', 'periodic'], default='random')
    args = parser.parse_args()
    build_path = TARGET / 'half-bench-build.json'
    if args.phase in ('build', 'all'):
        run(['python3', PORT / 'verify_half.py'])
        report = json.loads((PORT / 'half-results.json').read_text())
        report['source_sha256'] = hashes()
        report['binary_sha256'] = {p: hashlib.sha256((TARGET / p).read_bytes()).hexdigest()
                                   for p, _ in set(CANDIDATES.values())}
        build_path.write_text(json.dumps(report, indent=2) + '\n')
    if args.phase == 'build':
        print('Exhaustive conversion and reduction parity checked; benchmark ready.')
        return
    report = json.loads(build_path.read_text())
    assert report['source_sha256'] == hashes(), 'Sources changed; rebuild'
    for p, expected in report['binary_sha256'].items():
        assert hashlib.sha256((TARGET / p).read_bytes()).hexdigest() == expected
    report.update({'measurement_utc': datetime.now(timezone.utc).isoformat(),
                   'platform': platform.platform(), 'input_pattern': args.pattern,
                   'scope': 'Native conversion+selection versus safe fused half reduction; hostcopy means host allocation+copy, never GPU/device readback. No FFI/full-engine/model integration.',
                   'timing_status': 'Provisional shared-cloud-host microbenchmarks; known project workers idle, no affinity/thermal/frequency control. GCC versus Rust LLVM.',
                   'samples': []})
    for vocab in (32768, 131072, 262144):
        for batch, sequence in ((1, 1), (4, 1), (1, 8), (4, 8)):
            rounds = max(2, 2_000_000 // (vocab * batch * sequence))
            samples = {k: [] for k in CANDIDATES}
            checksums = set()
            for repeat in range(7):
                order = list(CANDIDATES)
                order = order[repeat % len(order):] + order[:repeat % len(order)]
                for name in order:
                    binary, mode = CANDIDATES[name]
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
