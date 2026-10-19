#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Interleave original and current Rust benchmarks on this machine."""
import io
import json
import platform
import hashlib
from pathlib import Path
import statistics
import subprocess
import tarfile

from verify import PORT, ROOT, TARGET, run

ORIGINAL = "c7f5c7b7c4a7bc4ff5ffd036c33ed71cdf40d509"


def main():
    source = TARGET / "original-source"
    source.mkdir(parents=True, exist_ok=True)
    archive = subprocess.check_output(["git", "archive", ORIGINAL, "rust-port"], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(source, filter="data")
    run(["cargo", "build", "--release", "--locked", "--offline", "--manifest-path",
         source / "rust-port/Cargo.toml", "--target-dir", TARGET / "original-build"])
    run(["cargo", "build", "--release", "--locked", "--offline", "--manifest-path", PORT / "Cargo.toml"])
    original = TARGET / "original-build/release/stop-trace"
    current = TARGET / "release/stop-trace"
    report = {"original_commit": ORIGINAL,
              "current_commit": run(["git", "rev-parse", "HEAD"]).strip(),
              "current_source_sha256": hashlib.sha256((PORT / "src/lib.rs").read_bytes()).hexdigest(),
              "platform": platform.platform(),
              "rustc": run(["rustc", "--version"]).strip(),
              "comparison": {}}
    for mode in ["no-match", "partial", "early-stop"]:
        samples = {"original": [], "current": []}
        checksums = set()
        for repeat in range(11):
            order = [("original", original), ("current", current)]
            if repeat % 2:
                order.reverse()
            for name, binary in order:
                ns, checksum = map(int, run([binary, "bench", mode, 20_000]).split())
                checksums.add(checksum)
                if repeat >= 2:
                    samples[name].append(ns / (20_000 * 64))
        assert len(checksums) == 1
        report["comparison"][mode] = {
            "samples_ns_per_batch_step": samples,
            "median_ns_per_batch_step": {k: statistics.median(v) for k, v in samples.items()},
            "checksum": checksums.pop(),
        }
    (PORT / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
