# Cloud port checkpoint (2026-10-19)

Continuation now runs exclusively on the dot cloud computer. The earlier local
checkpoint is retained below as history, not a request to resume local work.

- Official Rust toolchains, CMake and Abseil were installed only in the cloud
  workspace. The crate passes its tests on Rust 1.70.0 and 1.99.1; the lockfile
  format supports both. It retains zero Cargo dependencies and forbids unsafe.
- Stop detection passes 452 C++ traces / 41,135 operations. Serial cloud A/B
  medians improve by 17.9%, 22.8%, 7.4% for no-match/partial/early-stop versus
  the first Rust milestone. Raw samples and environment are in benchmarks/.
- The pure-Rust greedy k=1 candidate is implemented and exported. Nineteen new
  unit tests and 11,481 C++ differential cases cover shape checks and exact
  float-comparison ordering. No general sampling or native integration yet.
- All validation commands and limitations are documented in README.md. Set
  PKG_CONFIG_PATH for a non-system Abseil install. Run benchmarks serially.

Next bounded work: port the detector nested single-token-vector overload and
validate it against unchanged native code, then audit its actual generation-task
integration boundary. Do not mistake standalone component parity for an engine
port, native build integration, or end-to-end model inference validation.

# Historical migration checkpoint

Local port development stopped at the user's request. No builds, benchmarks or
Claude jobs remain running from this task. Continue in a separate cloud checkout;
do not resume heavy work on Omarchy.

The first port milestone is commit `c7f5c7b7c4a7bc4ff5ffd036c33ed71cdf40d509`.
The next checkpoint changes only Rust stop-detector iteration and reset storage,
and adds `compare.py` to build that first milestone and compare it with the
current implementation in alternating order.

Completed before the stop request: the optimized detector passed the existing
five unit tests, 452 C++ differential traces containing 41,135 operations, and
clippy with warnings denied. Assembly inspection found seven indirect
bounds-panic call sites in the original process function and none in the updated
function. No hardware profiling tools were installed locally.

Rerun performance measurements in cloud before drawing conclusions. The local
optimization verification and A/B comparison overlapped, so their timing samples
are provisional. `benchmarks/baseline.json` remains the original recorded run.
Run verification and comparison serially, with no competing benchmark job.

The next candidate selected from the runtime architecture is the greedy `k=1`
path in `runtime/components/sampling_cpu_util.cc`, called by TopKTopPSampling.
It reduces logits using 128-element tiles with eight maximum lanes. A Rust port
must preserve first-index tie behavior and the exact C++ comparison ordering for
NaNs; Rust's float max operation has different NaN semantics. Validate vocabulary
boundaries around 128, batch/sequence shapes, ties, infinities and NaNs against
the unchanged native source. No code for this candidate has been written yet.

The detector's nested single-token-vector overload is also still unported. The
generation task in `runtime/core/tasks.cc` uses that overload when withholding
partial stop sequences and trimming completed matches. Current Rust code remains
standalone, with no native runtime or Mobie integration.

Do not resend local source or test harnesses to Claude until the parent supplies
new explicit authorization addressing the earlier source-disclosure rejection.
No alternate transmission route was attempted.
