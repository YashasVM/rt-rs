#!/usr/bin/env python3
# Copyright 2026 rt-rs contributors.
# SPDX-License-Identifier: Apache-2.0
"""Profile pinned native greedy sampling versus its reduction and the Rust port.

Build/parity and timing are separable so shared-host quiet windows need no build.
Shapes are representative workload sizes, not a claim about a tested model.
"""
import argparse
import hashlib
import json
import os
import platform
import shlex
import statistics
from datetime import datetime, timezone

from verify import CURRENT_MAIN, PORT, ROOT, TARGET, run

SOURCES = ['src/greedy.rs', 'src/bin/greedy_trace.rs', 'tests/greedy_reference.cc',
           'benchmark_greedy.py']
BINARIES = {'cpp_reduction': TARGET / 'greedy-reference',
            'cpp_sampling': TARGET / 'greedy-reference',
            'rust_reduction': TARGET / 'release/greedy-trace'}


def hashes():
    return {name: hashlib.sha256((PORT / name).read_bytes()).hexdigest() for name in SOURCES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['build', 'run', 'all'], default='all')
    args = parser.parse_args()
    build_path = TARGET / 'greedy-bench-build.json'
    if args.phase in ('build', 'all'):
        env = dict(os.environ, RT_RS_CURRENT_MAIN='1')
        run(['python3', PORT / 'verify_greedy.py'], env=env)
        cpp = os.environ.get('CXX', 'c++')
        flags = shlex.split(run(['pkg-config', '--cflags', '--libs', '--static',
                                'absl_status', 'absl_statusor', 'absl_strings', 'absl_str_format']))
        run([cpp, '-std=c++20', '-O3', '-DNDEBUG', '-DRT_RS_COUNT_ALLOCATIONS',
             '-I', ROOT, PORT / 'tests/greedy_reference.cc',
             ROOT / 'runtime/components/sampling_cpu_util.cc',
             '-o', TARGET / 'greedy-allocations', *flags])
        build = {'baseline': CURRENT_MAIN, 'source_sha256': hashes(),
                 'rustc': run(['rustc', '--version']).strip(),
                 'cxx': run([cpp, '--version']).splitlines()[0],
                 'binary_sha256': {str(p.relative_to(PORT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in set(BINARIES.values()) | {TARGET / 'greedy-allocations'}}}
        build_path.write_text(json.dumps(build, indent=2) + '\n')
    if args.phase == 'build':
        print('Parity checked; timing binaries prepared.')
        return
    report = json.loads(build_path.read_text())
    assert report['source_sha256'] == hashes(), 'Sources changed; rebuild before timing'
    for name, expected in report['binary_sha256'].items():
        assert hashlib.sha256((PORT / name).read_bytes()).hexdigest() == expected
    report.update({'measurement_utc': datetime.now(timezone.utc).isoformat(),
                   'platform': platform.platform(),
                   'scope': 'Finite deterministic logits; full native k=1 sampling versus native/Rust reduction. Not equivalent complete output APIs, not model inference or native-to-Rust integration.',
                   'timing_status': 'Provisional shared-host timings; no affinity, thermal or frequency control.',
                   'samples': [], 'allocations': []})
    for vocab in (32768, 131072, 262144):
        for batch, sequence in ((1, 1), (4, 1), (1, 8), (4, 8)):
            rounds = max(4, 8_000_000 // (vocab * batch * sequence))
            samples = {name: [] for name in BINARIES}
            checksums = set()
            for repeat in range(7):
                order = list(BINARIES)
                # Rotate starting candidate to reduce consistent ordering bias.
                order = order[repeat % 3:] + order[:repeat % 3]
                for name in order:
                    mode = 'sampling' if name == 'cpp_sampling' else 'reduction'
                    ns, checksum = map(int, run([BINARIES[name], mode, batch, sequence, vocab, rounds]).split())
                    checksums.add(checksum)
                    if repeat >= 2:
                        samples[name].append(ns / (rounds * batch * sequence))
            assert len(checksums) == 1, 'Greedy indices differ'
            row = {'vocab': vocab, 'batch': batch, 'sequence': sequence, 'rounds': rounds,
                   'checksum': checksums.pop(), 'ns_per_position_samples': samples,
                   'ns_per_position_median': {k: statistics.median(v) for k, v in samples.items()}}
            report['samples'].append(row)
            print(vocab, batch, sequence, row['ns_per_position_median'], flush=True)
    # Separate instrumented binary: no allocation-counter overhead in timings.
    for batch, sequence in ((1, 1), (4, 1), (1, 8), (4, 8)):
        for mode in ('reduction', 'sampling'):
            for rounds in (1, 100):
                count, size, checksum = map(int, run([TARGET / 'greedy-allocations', mode,
                                                     batch, sequence, 32768, rounds]).split())
                report['allocations'].append({'mode': mode, 'batch': batch, 'sequence': sequence,
                                              'vocab': 32768, 'calls': rounds, 'new_calls': count,
                                              'requested_bytes': size, 'checksum': checksum})
    (PORT / 'greedy-benchmark-results.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
