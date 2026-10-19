# FP16 greedy integration gates

Scope: source audit of LiteRT-LM v0.18.0 (`b2f686e2ed4718fb84ec398a61dd59ca0f0aff27`)
and pinned main (`e32526ccbc54fed8d7ba8fbf292c03572fba5081`). This is not an
end-to-end measurement. The final section describes a scoped, host-tested
Bazel sampler integration. The original sampler .cc, .h and test .cc are byte-identical
between these two upstream pins.

## Actual native path

`runtime/components/top_p_cpu_sampler.cc::SampleToIdAndScoreBuffer` validates
input/output tensors, reads FP16 data into a temporary host `vector<uint16_t>`,
converts the whole shape into retained `logits_data_` float storage, invokes
`TopKTopPSampling`, flattens returned IDs, and writes them to a TensorBuffer.
Device readback/synchronization and output writes are absent from our benchmark.
The optional host-copy benchmark is not a substitute for measuring them.

`sampling_cpu_util.cc::TopKTopPSampling` validates nonempty logits, divisible
shape, positive k, p range, and nonnull RNG. It selects top-k, allocates flattened
indices and softmax intermediates, then for k=1 returns selected IDs and scores
of exactly 1.0 without drawing from the RNG. Softmax rejects negative temperature
before that return. The reduction-only API deliberately has a different empty
vocabulary contract, so its parity cannot establish sampler-level parity.

## Safe applicability boundary

A future fused greedy branch must be limited to effective k=1, retain existing
validation/error behavior, and preserve all selected-index NaN/tie ordering.
It must not silently reinterpret temperature=0 as permission to change k.
General top-k/top-p, RNG sequencing and sampling quality are outside this port.

When a scores tensor is supplied and `compute_exact_log_probs_` is true,
`ComputeExactLogProbsForBatch` traverses the full float32 logits, including its
own nonfinite handling, nominal temperature for greedy, double accumulation,
and final clamping. The current fused ID-only reduction does not implement this
operation. Keep the original conversion/scoring path in that mode until an
independently validated equivalent exists. When approximate scores are requested
for k=1, native emits log(1.0)=0.0; output shape validation and write behavior
still need preservation. No score API is promised by the standalone adapter.

## Required evidence before enabling a production option

- Build and test the actual sampler and all consumers, not just standalone ABI.
- Verify archive inclusion and ABI/layout propagation in the real final link.
- Preserve TensorBuffer ownership, lifetime, synchronization, error propagation,
  and cancellation behavior. Do not assume host pointers for device buffers.
- Differential-test complete sampler status, IDs, scores, and RNG state across
  supported configurations, including invalid metadata and edge float values.
- Profile real model tensor shapes/readback and whole token latency on the target
  device, with matched compiler/optimization settings where feasible.
- Keep the original native fallback and avoid a default switch before those gates.

Current component timings use GCC for C++ and Rust's LLVM backend on a shared
x86_64 cloud host without affinity/thermal controls. They motivate a bounded
boundary experiment, not a claim that Rust or LiteRT-LM inference is faster.

## Hosted baseline experiment

`.github/workflows/rust-sampler-baseline.yml` runs the existing real
`//runtime/components:top_p_cpu_sampler_test` on a standard Ubuntu 24.04 hosted
runner, bounded to 60 minutes, two jobs and 6144 MB Bazel local RAM resources.
It uses supported LiteRT `build_include=cpu_only`, not runtime stubs. The original baseline
trigger covered only workflow changes; the integration matrix also watches its
relevant sampler/Rust source paths on the existing Rust branch. Checkout is SHA-pinned,
credentials are not persisted, permissions are contents-read only, and no models,
secrets, remote caches, artifacts or larger/paid runners are requested.

The preinstalled official runner Bazelisk honors repository `.bazelversion`
7.6.1; dependency archive pins remain upstream's WORKSPACE pins. The hosted OS,
Clang and launcher are image-managed rather than hermetically pinned, so actual
versions are printed. Standard public-repository runner execution is free per
GitHub documentation; repository visibility was verified public before setup.
Its documented 14 GB SSD does not guarantee this dependency build will fit.
The test reaches LiteRT CPU runtime implementation transitively despite needing
no model. Success/failure must be reported, not hidden with alternate stubs.
The unchanged baseline passed 12/12 tests in run 39326962690 at commit 67c07f1.
Bazel elapsed 2038.37s; output 7.1GB; observed free disk 85GB before/78GB after,
with Clang 18.1.3 and Bazel 7.6.1. The raw concise evidence is retained in
`benchmarks/hosted-native-sampler-baseline.json`. It is not model inference.

The opt-in follow-up uses `--define=rt_rs_fp16_greedy=true` and explicit
`top_p_cpu_sampler -> greedy_rust -> rt_rs_stop_ffi -> rt_rs_stop` Bazel deps.
Only sampler.cc receives the local define; no public class/header layout changes.
Default native and enabled paths each run paired real-TensorBuffer parity tests
on separate bounded standard runners. Binary symbol checks require the FP16 Rust
entry point only in the enabled binary; symbol presence proves linkage, not
invocation counts. Independent source review found no blocking issue in the
hook, dependency graph, seven added paired tests, or matrix. Both hosted legs
passed all 19 real TensorBuffer tests and linkage checks in run 39332219521
at d50e3bf. The concise evidence is in `benchmarks/hosted-sampler-integration.json`.
No CMake option, model inference, or final engine link is claimed by that result.

Sources checked 2026-10-19:
- https://docs.github.com/en/actions/reference/runners/github-hosted-runners
- https://github.com/actions/runner-images/blob/main/images/ubuntu/Ubuntu2404-Readme.md
- https://github.com/actions/runner-images/blob/main/images/ubuntu/scripts/build/install-bazel.sh


## Final engine linkage gate

The next separate hosted workflow builds `//runtime/engine:litert_lm_main`
with the same supported CPU-only settings and explicit Rust FP16 option. This
adds actual tokenizer, llguidance, template and parser Rust consumers absent
from sampler tests. Mixed Rust staticlib/rlib runtime linkage must succeed
without duplicate-symbol suppression, tokenizer removal, whole-archive changes
or stubs. The same pinned Bazel toolchain supplies all Rust components.

A defined FP16 Rust symbol and upstream static-LiteRt dependency check are
followed by a bounded help-only startup smoke. Help must expose the expected
model/backend flags; no model is downloaded or loaded. Success would establish
final executable linkage/startup, not engine creation, tokenization, FP16 model
execution or speed. This next result is pending. Android, CMake and actual
model/device performance remain separate gates.
