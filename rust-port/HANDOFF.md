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
the original detector headers and build graphs remain unchanged. Future detector integration
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
The later stateless ABI and opt-in sampler checkpoints below supersede the
initial safe-only scope. Real device/model measurements remain necessary
before any inference performance claim.

## FP16 native boundary checkpoint

The standalone stateless binary16 ABI and owned C++ wrapper now pass 10,995
native differential traces, nine ABI tests, and normal/sanitized native contracts.
Independent unsafe review found no blocker in the standalone boundary; native
defaults remain unchanged.
Equivalent GCC fused control is retained alongside original conversion+selection
and prior slower Rust implementation. Boundary gains persist across random,
ascending, and periodic profiles, with exact per-distribution raw samples and
compiler/copy-cost caveats in README and benchmarks. Do not generalize to Android
or inference. `SAMPLER_INTEGRATION.md` records actual TensorBuffer, validation,
RNG and exact-log-probability gates before a future engine option.

## Real TensorBuffer sampler gate

Unchanged hosted native baseline passed 12/12 tests in run 39326962690, with
2038.37s cold build and 7.1GB output. The opt-in Bazel sampler flag now has an
explicit Rust staticlib dependency and .cc-local define, no layout/CMake change.
Both hosted legs passed all 19 paired real TensorBuffer tests and correct
binary linkage checks at d50e3bf, run 39332219521. Cold build times were
1091.57s native and 1462.66s Rust on separate runners; these are not comparable
runtime-performance measurements. The real engine link/help gate is next.
Exact scoring, non-k1 and invalid-config paths retain native fallback. Do not
claim full-engine linkage, device readback latency or inference speed.

## Engine dependency materialization gate

Run 39335660149 reached final link but found an unmaterialized upstream Git LFS
constraint-provider .so. This is not a resource-limit failure or proof of a Rust
link error. Preserve the failure evidence and retrieve only the original pinned,
hash/size-verified upstream binary before rerunning the unchanged engine gates.
No final linkage/help/inference result is yet established. Future phone/model
validation is specified in DEVICE_VALIDATION.md, not performed by that document.

## Final Linux engine checkpoint

Pinned original-object materialization fixed the LFS failure. Run 39339048193
at 58f83cce passed genuine CPU-only opt-in engine linkage, Rust symbol presence,
static-LiteRT dependency check and bounded help startup. Build: 4971 actions,
1469.080s, 7.6 GB output; executable 30,598,032 bytes. No link suppression or
consumer removal. Same-head Rust checks passed. This is not model inference,
sampling invocation-count, Android or speed evidence. Next use the documented
paired device/model protocol once the relevant artifacts/access are approved.
