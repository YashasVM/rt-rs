# FP16 greedy integration gates

Scope: source audit of LiteRT-LM v0.18.0 (`b2f686e2ed4718fb84ec398a61dd59ca0f0aff27`)
and pinned main (`e32526ccbc54fed8d7ba8fbf292c03572fba5081`). This is not an
engine integration or an end-to-end measurement.

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
