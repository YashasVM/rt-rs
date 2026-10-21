#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit pinned Git objects without fetching, checking out, or building code.

Print JSON to stdout. Exit nonzero for missing objects or changed audited files
between baseline and latest main. Historical differences are informational.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
BASELINE = "b2f686e2ed4718fb84ec398a61dd59ca0f0aff27"
HISTORICAL_MAIN = "e32526ccbc54fed8d7ba8fbf292c03572fba5081"
LATEST_MAIN = "13c208f1cf401bb024bd993f1debb00e90fc5bfb"
FORK_CHECKPOINT = "19ec82ece59abd3af2e21962a1c2033f7fc6dadf"
PINS = {
    "baseline": BASELINE,
    "historical_main": HISTORICAL_MAIN,
    "latest_main": LATEST_MAIN,
}
PATHS = [
    "runtime/components/stop_token_detector.h",
    "runtime/components/stop_token_detector.cc",
    "runtime/components/sampling_cpu_util.h",
    "runtime/components/sampling_cpu_util.cc",
    "runtime/components/top_p_cpu_sampler.cc",
]


def git(*args):
    # Disable implicit fetching in partial clones as well as optional locks.
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_OPTIONAL_LOCKS="0")
    result = subprocess.run(
        ["git", "--no-replace-objects", *args], cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Required local Git object unavailable or unreadable ({' '.join(args)}): "
            f"{detail}. No fetch, checkout, or rebase was attempted."
        )
    return result.stdout


def audit():
    for commit in [*PINS.values(), FORK_CHECKPOINT]:
        git("cat-file", "-e", f"{commit}^{{commit}}")
    files = []
    for path in PATHS:
        contents = {name: git("cat-file", "blob", f"{pin}:{path}")
                    for name, pin in PINS.items()}
        native_diff = git("diff", "--no-ext-diff", "--no-textconv", "--binary",
                          BASELINE, LATEST_MAIN, "--", path)
        historical_diff = git("diff", "--no-ext-diff", "--no-textconv", "--binary",
                              HISTORICAL_MAIN, LATEST_MAIN, "--", path)
        files.append({
            "path": path,
            "sha256": {name: hashlib.sha256(data).hexdigest()
                       for name, data in contents.items()},
            "byte_lengths": {name: len(data) for name, data in contents.items()},
            "latest_equals_baseline": contents["latest_main"] == contents["baseline"],
            "latest_equals_historical_main": contents["latest_main"] == contents["historical_main"],
            "baseline_to_latest_git_diff_empty": not native_diff,
            "historical_to_latest_git_diff_empty": not historical_diff,
        })
    ahead, behind = map(int, git("rev-list", "--left-right", "--count",
                                f"{FORK_CHECKPOINT}...{LATEST_MAIN}").split())
    return {
        "audit_date": "2026-10-21",
        "audit_kind": "pinned Git source identity; not a fresh C++/Rust parity run",
        "pins": PINS,
        "fork_checkpoint": FORK_CHECKPOINT,
        "fork_vs_latest_main": {"ahead": ahead, "behind": behind},
        "scope": "Only the five listed native files, not whole-runtime compatibility",
        "all_latest_files_identical_to_baseline": all(
            item["latest_equals_baseline"] and item["baseline_to_latest_git_diff_empty"]
            for item in files
        ),
        "files": files,
        "rollback_contract": {
            "historical_extension": "Singleton rollback marking remains a Rust API extension",
            "latest_main": "Historical singleton rollback marking is absent; detector equals baseline",
            "historical_verifier_pin_unchanged": HISTORICAL_MAIN,
        },
        "validation": "Read-only local Git object inspection; no fetch, rebase, build, or benchmark",
    }


def main():
    try:
        report = audit()
    except RuntimeError as error:
        print(f"Upstream audit failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0 if report["all_latest_files_identical_to_baseline"] else 1


if __name__ == "__main__":
    sys.exit(main())
