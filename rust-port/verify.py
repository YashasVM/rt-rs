#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build and compare native Rust against the unmodified pinned C++ detector."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import random
import shlex
import statistics
import subprocess

PORT = Path(__file__).resolve().parent
ROOT = PORT.parent
BASELINE = "b2f686e2ed4718fb84ec398a61dd59ca0f0aff27"
LITERT_BASELINE = "3e49a3789864362f6069e0568435b584a38cf917"
LITERT = Path(os.environ.get("LITERT_SOURCE_DIR", ROOT.parent / "litert-reference"))
TARGET = PORT / "target"


def run(args, **kwargs):
    return subprocess.check_output([str(a) for a in args], text=True, cwd=ROOT, **kwargs)


def main():
    for path in ["runtime/components/stop_token_detector.cc",
                 "runtime/components/stop_token_detector.h",
                 "runtime/util/status_macros.h", "support/util/status_macros.h"]:
        expected = run(["git", "show", f"{BASELINE}:{path}"])
        if (ROOT / path).read_text() != expected:
            raise SystemExit(f"Reference source differs from pinned baseline: {path}")
    if run(["git", "-C", LITERT, "rev-parse", "HEAD"]).strip() != LITERT_BASELINE:
        raise SystemExit("Set LITERT_SOURCE_DIR to the documented pinned LiteRT checkout")
    # Also reject local changes to the dependency headers.
    run(["git", "-C", LITERT, "diff", "--exit-code", "HEAD", "--",
         "litert/c", "litert/cc", "litert/build_common"])
    run(["cargo", "test", "--manifest-path", PORT / "Cargo.toml", "--locked", "--offline"])
    run(["cargo", "build", "--release", "--manifest-path", PORT / "Cargo.toml", "--locked", "--offline"])
    cpp = os.environ.get("CXX", "c++")
    libs = shlex.split(run(["pkg-config", "--cflags", "--libs", "--static", "absl_status", "absl_statusor",
                           "absl_log", "absl_check", "absl_strings", "absl_str_format"]))
    reference = TARGET / "reference"
    config_dir = TARGET / "include/litert/build_common"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = (LITERT / "litert/build_common/build_config.h.in").read_text()
    for option in ["LITERT_BUILD_CONFIG_DISABLE_GPU", "LITERT_BUILD_CONFIG_DISABLE_NPU"]:
        config = config.replace(f"#cmakedefine01 {option}", f"#define {option} 1")
    (config_dir / "build_config.h").write_text(config)
    run([cpp, "-std=c++20", "-O3", "-DNDEBUG", "-I", TARGET / "include", "-I", ROOT, "-I", LITERT,
         PORT / "tests/reference.cc", ROOT / "runtime/components/stop_token_detector.cc",
         "-o", reference, *libs])
    rust = TARGET / "release/stop-trace"
    traces = [
        "N 2\nA 5\nA 7 8 9\nP 3 1\nP 6 0\nP 7 0\nP 8 0\nP 9 0\nP 10 6\nP 11 5\nR 0\nP 5 5\n",
        "N 1\nA 1 1 2\nP 1\nP 1\nP 1\nP 2\nA 1 1 2\nA\nP\nR 3\nP 1\nP 1 1 1\n",
        "N 1\nA 1 2\nA 2\nP 1\nP 2\nP -1\nR 0\nP 1\nA 1 2 3\nP 2\nP 3\n",
        "N 1\nP 7\nA 7 8\nP 8\nA 5\nA 3 4 5 6\nP 3\nP 4\nP 5\nP 6\nA 6\nP 0\n",
    ]
    # Exhaust all streams of length 6 over {0,1} against overlapping sequences.
    for seq in [(0,), (0, 0), (0, 1), (0, 0, 1), (0, 1, 0)]:
        for stream in itertools.product(range(2), repeat=6):
            traces.append("N 1\nA " + " ".join(map(str, seq)) + "\n" +
                          "".join(f"P {token}\n" for token in stream))
    rng = random.Random(1800)
    for _ in range(128):
        batch = rng.randint(1, 4)
        commands = [f"N {batch}"]
        sequences = []
        for _ in range(300):
            action = rng.random()
            if action < .18:
                seq = (rng.choice(sequences) if sequences and rng.random() < .25
                       else [rng.randint(-2, 3) for _ in range(rng.randint(0, 5))])
                commands.append("A " + " ".join(map(str, seq)))
                sequences.append(seq)
            elif action < .24:
                size = rng.randint(0, 4)
                batch = size or batch
                commands.append(f"R {size}")
            else:
                size = batch if rng.random() < .9 else batch + 1
                commands.append("P " + " ".join(str(rng.randint(-2, 3)) for _ in range(size)))
        traces.append("\n".join(commands) + "\n")
    all_input = "".join(traces)
    expected = run([reference], input=all_input)
    actual = run([rust], input=all_input)
    if expected != actual:
        (TARGET / "failed-trace.txt").write_text(all_input)
        for i, (a, b) in enumerate(zip(expected.splitlines(), actual.splitlines())):
            if a != b:
                raise AssertionError(f"Mismatch after operation {i}: C++={a}, Rust={b}")
        raise AssertionError("Trace output length mismatch")
    report = {
        "baseline": BASELINE,
        "litert_headers": LITERT_BASELINE,
        "platform": platform.platform(),
        "rustc": run(["rustc", "--version"]).strip(),
        "cxx": run([cpp, "--version"]).splitlines()[0],
        "absl": run(["pkg-config", "--modversion", "absl_status"]).strip(),
        "trace_count": len(traces),
        "operations": len(all_input.splitlines()),
        "trace_sha256": hashlib.sha256(all_input.encode()).hexdigest(),
        "benchmarks": {},
    }
    rounds = 20_000
    for mode in ["no-match", "partial", "early-stop"]:
        samples = {"cpp": [], "rust": []}
        checksums = set()
        for repeat in range(9):
            order = [("cpp", reference), ("rust", rust)]
            if repeat % 2:
                order.reverse()
            for name, binary in order:
                ns, checksum = map(int, run([binary, "bench", mode, rounds]).split())
                checksums.add(checksum)
                if repeat >= 2:
                    samples[name].append(ns / (rounds * 64))
        assert len(checksums) == 1, f"Benchmark state differs: {mode}"
        report["benchmarks"][mode] = {
            "ns_per_batch_step_samples": samples,
            "ns_per_batch_step_median": {k: statistics.median(v) for k, v in samples.items()},
            "checksum": checksums.pop(), "batch_size": 4,
            "sequences": [[7, 8, 9], [4, 4, 5], [99]],
            "rounds": rounds, "steps_per_round": 64,
        }
    (PORT / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
