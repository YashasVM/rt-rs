# Technical continuation checkpoint (2026-10-19)

## Implemented components

- Safe, dependency-free Rust stop-token detector, flat and nested singleton-row
  APIs. Nested validation is atomic and avoids a temporary flatten allocation.
- Current-main singleton rollback marks and per-step rollback flags. These only
  expose state; no executor/KV-cache operation is performed by this crate.
- Pure-Rust greedy k=1 logits selection, preserving eight-lane, 128-element tile
  comparison ordering including NaNs. General top-k/top-p sampling is not ported.

## Validation

Run the commands in README.md with Rust 1.70 or newer and the pinned native
references. Run benchmarks serially. Cloud benchmark JSON records include raw
samples and toolchain/platform details; do not compare timings across machines.
The Rust-only CI does not replace native parity or end-to-end device testing.

## Integration boundary

The current native generation path constructs StopTokenStreamFilter in
runtime/core/tasks.cc. It first updates the detector, then buffers potential
partial matches and suppresses completed stop sequences before detokenization.
Merely swapping token detection would not port this buffering/ownership layer.
Current-main additionally uses rollback flags to keep selected stop tokens out
of KV cache. Real integration must preserve both contracts and cancellation,
callback and executor lifetimes, including multi-candidate behavior.

Next: profile an actual native generation workload and measure allocations/time
at the detector/filter boundary before choosing an integration design. A model,
working native build and device/host baseline are needed to demonstrate any
end-to-end improvement. Keep the native path available as a reference/fallback.
No Mobie source or native runtime build graph has been modified by this port.

## Standalone adapter checkpoint

An isolated C ABI and C++ adapter now exercise the safe detector from native
code. They are not wired into any engine target. Keep this distinction explicit:
all production headers and build graphs remain unchanged. Future integration
must validate class-layout propagation to every consumer and the flattened final
archive linkage, not merely link the standalone contract executable. See README
for ownership, pointer, panic and exception contracts and reproduction commands.

The process-plus-snapshot experiment retains eager reference observability and
supports getter-input aliasing by ending all input borrows before export writes.
Its measured boundary still loses to native C++ in all six modes. Stop tuning
this tiny component for now; next profile greedy selection at realistic
vocabulary sizes while retaining exact NaN/tie semantics and allocation costs.

## Greedy native boundary

A separate stateless greedy adapter now validates native index ranges and buffer
metadata before exporting owned indices. It retains no pointers and does not
select any production sampler target. Cross-language parity covers 11,481 cases.
Read-only unsafe review found no remaining blocker after concurrency contracts
were made explicit; native C++ sanitizer contracts pass (Rust/Abseil not
instrumented). Periodic/random/ascending/descending profiles preserve all raw
samples and GCC-versus-LLVM caveats. Results are component evidence, not Android
or inference gains. Next profile native FP16 conversion plus selection before
attempting a safe fused path that avoids whole-vocabulary float32 staging.

## FP16 prototype

Safe half conversion matches all 65,536 native bit encodings; fused greedy matches
10,995 native traces, with 34 core unit tests passing. The initial branch-heavy
order-key implementation is slower on most random-bit workloads despite avoiding
float32 staging; its negative report and baseline tree remain reproducible.
The independently reviewed branchless equivalent wins all tested shapes versus
that old Rust variant. Native conversion-plus-selection median ratios range
1.63–6.32x across three distributions, or 1.50–3.86x with equal host-copy costs.
Keep per-distribution samples and GCC-versus-LLVM/shared-host caveats visible.
No FP16 unsafe ABI or engine integration exists. A bounded stateless native
boundary experiment, with independent safety review and equivalent full-path
validation, is the next justified step; real tensor readback/device/model
measurements remain necessary before any runtime performance claim.

## FP16 native boundary checkpoint

The standalone stateless binary16 ABI and owned C++ wrapper now pass 10,995
native differential traces, nine ABI tests, and normal/sanitized native contracts.
Independent unsafe review found no blocker; production targets remain unchanged.
Equivalent GCC fused control is retained alongside original conversion+selection
and prior slower Rust implementation. Boundary gains persist across random,
ascending, and periodic profiles, with exact per-distribution raw samples and
compiler/copy-cost caveats in README and benchmarks. Do not generalize to Android
or inference. `SAMPLER_INTEGRATION.md` records actual TensorBuffer, validation,
RNG and exact-log-probability gates before a future engine option.
