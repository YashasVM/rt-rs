# Native versus opt-in Rust: device validation protocol

## Status and scope (2026-10-19)

This is a future experiment, not a device result or a deployment instruction.
The native default is unchanged. The option replaces only eligible FP16 CPU
sampling, not the LiteRT executor or the whole runtime. The real TensorBuffer
sampler tests passed 19/19 with the option OFF and ON in hosted Linux run
39332219521. The separate final CPU engine run 39335660149 failed at link because an
upstream shared library was still a Git LFS pointer; its checked-object repair
and final link/help retry are pending. Even a successful help smoke is not inference.
See [integration evidence and boundaries](SAMPLER_INTEGRATION.md).

Evidence currently covers Bazel/Linux only. CMake, Android artifacts, final
Android linking, ABI compatibility, and execution on a phone are separate,
unpassed gates. For an app, JNI/library packaging and loading must also pass;
a command-line binary does not validate JNI. No phone access, model download,
license acceptance, installation, or deployment is performed or authorized by
this document. Obtain the user's approved device and lawfully available model
before those steps.

## 1. Freeze the comparison

Create a run manifest before collecting results:

- Source: upstream v0.18.0 `b2f686e2ed4718fb84ec398a61dd59ca0f0aff27`;
  audited upstream main `e32526ccbc54fed8d7ba8fbf292c03572fba5081`;
  working branch `rust/stop-token-detector-v0.18.0`, inspected commit
  `d34276697f69eecca2c5b16a8e43816ce860fcd8`. Record the actual tested commit,
  clean/dirty status and complete patch, including any harness instrumentation.
- Build two binaries from that SAME revision and harness: native OFF versus
  `--define=rt_rs_fp16_greedy=true`. Record full build commands, configurations,
  optimization/LTO/target CPU flags, dependency pins, executable and library
  SHA-256 hashes. Do not compare differently optimized or different revisions.
- Record actual Bazel/Bazelisk, Rust target toolchain and rustc version, Clang,
  linker, libc/C++ runtime, Java, and host OS versions. `.bazelversion` is 7.6.1;
  WORKSPACE supplies rules_rust/toolchains, not the ambient Cargo compiler.
  Android additionally needs exact NDK/SDK/API level, target triple and ABI,
  packaging/JNI version if applicable, and loader/dependency inspection.
- Model: exact filename, source/version, SHA-256, architecture, quantization,
  context limit and actual logits dtype/shape. Hash tokenizer assets and chat
  templates/configuration too; if embedded, record the enclosing model hash
  and embedded asset identity. Identical model weights do not prove FP16 logits.
- Device: model/SoC, RAM, architecture/ABI, OS/build/kernel, backend and driver
  versions, thread count, affinity, power/charging mode and background load.
  Fix main executor AND sampler backends. CPU-only hosted evidence does not
  transfer to GPU/NPU; measure each supported combination separately.

## 2. Build, link and path gates

The existing Linux engine workflow provides this verified command pattern
(repository root; prerequisites from the upstream build guide must exist):

```sh
# Run separately with MODE=false and MODE=true; preserve each resulting binary
# and build log before the next build overwrites bazel-bin.
MODE=false
CC=clang CXX=clang++ bazelisk build \
  --config=linux_x86_64 \
  --@litert//litert/build_common:build_include=cpu_only \
  --define=rt_rs_fp16_greedy="$MODE" \
  --jobs=2 --local_ram_resources=6144 --verbose_failures \
  //runtime/engine:litert_lm_main
```

This reproduces a build request, not a claim that the engine gate passed.
Repeat the real sampler tests with both modes and retain failures as well as
successes. Inspect final symbols/dependencies and loader startup. The Rust FP16
entry point must be present only in the enabled build. Symbol presence alone
does not prove that inference calls it.

For Android, first follow the repository's documented SDK/NDK configuration and
`--config=android_arm64` target setup, then establish OFF/ON build, final link,
correct ABI, packaging/loading and real sampler correctness on that target.
Do not assume adding the Linux define to an Android command yields a usable
artifact. Keep upstream Rust tokenizer/grammar consumers and link semantics;
no stubs, removed consumers or duplicate-symbol suppression to pass the gate.

Prove actual invocation using an untimed diagnostic build or profiler, recording
FP16 logits shape/dtype, effective parameters, fallback reason and invocation
counts. Use equivalent instrumentation for OFF/ON and disable it for timed runs.
Eligible ON calls require effective k=1, valid p and nonnegative temperature,
valid nonempty shape/RNG, and either no scores output or approximate scores.
Exact-log-probability output with a scores tensor takes the native fallback;
so do FP32 and non-greedy cases. Temperature=0 is not permission to change k.
Document fallback controls as such, not failed or accelerated Rust runs.

## 3. Use the actual runner, not merely accepted flags

Source inspection matters: `litert_lm_main.cc` consumes `--model_path`,
`--backend`, `--input_prompt`/`--input_prompt_file`, `--num_cpu_threads`,
`--enable_ynnpack` and `--visual_token_budget`. It runs one asynchronous
conversation and prints benchmark info. Shared flags can appear in help without
being wired into this main. In particular, the older build guide's benchmark,
async and iteration example is not sufficient for controlled benchmarking of
this current main.

`litert_lm_advanced_main.cc` and its BUILD target exist and consume
`--benchmark`, `--benchmark_prefill_tokens`, `--benchmark_decode_tokens`,
`--async`, `--num_iterations`, `--sampler_backend`,
`--report_peak_memory_footprint` and `--metric_proto_file_path`. Its separate
build/link and execution must be validated before use. The following is a
source-verified invocation TEMPLATE, not a completed or device-validated run;
BINARY and MODEL_PATH must identify approved, validated artifacts:

```sh
"$BINARY" --model_path="$MODEL_PATH" --backend=cpu --sampler_backend=cpu \
  --benchmark --benchmark_prefill_tokens=1024 --benchmark_decode_tokens=256 \
  --async=false --num_iterations=1 --report_peak_memory_footprint=true
```

Synthetic token-count benchmarking is not a natural-prompt quality test.
Neither inspected main exposes direct k/seed/temperature CLI overrides. Resolve
and log model metadata and effective SessionConfig parameters. If they cannot
be controlled by the selected runner, first implement and validate a paired
harness using `SessionConfig::GetMutableSamplerParams()` and
`SetSamplerBackend()`. Do not invent `--top_k` or `--seed` flags. Likewise, warm
engine reuse, exact token-ID capture and timestamp collection below are harness
requirements (pseudoprocedure), not capabilities proved by this command.

## 4. Paired quality and performance procedure

1. Predeclare a representative prompt corpus with exact UTF-8 bytes/hashes,
   chat template, token IDs/counts, short/long prefills and decode lengths,
   stop/EOS rules, context/cache policy and seeds. Fix effective k=1, p,
   temperature, score mode, output limits and all backend settings identically.
   Greedy selection should not consume RNG, but retain seed/state evidence.
2. Before timing, compare complete generated token IDs, decoded output,
   termination reason, statuses and requested scores between OFF and ON.
   Require exact greedy token/output parity on each matched case. Preserve
   approximate-score zero-log-probability checks and exact-score fallback tests.
   Run invalid-input, tie/nonfinite and fallback regression tests separately;
   record crashes, timeouts and errors, not just successful outputs. If the
   chosen backend is nondeterministic, resolve/isolate that before claiming
   exact parity. Text-only matching cannot establish token-ID equality.
3. Separate cold process start from warm engine requests. Define cold as a
   fresh process with explicitly recorded model/program/page-cache state;
   fresh process does not imply cold OS cache. Do not silently clear shared
   caches or require privileged cache drops. Warm means a loaded engine after
   declared warmups, with reset conversation/KV state unless testing a matched
   multi-turn workload. Verify runner iteration semantics before using them.
4. Measure monotonic process-launch-to-first-token (cold user-visible TTFT),
   request-submit-to-first-token (warm TTFT), first-to-last-token decode time,
   generated token count, full request wall time, model/engine load time and
   peak process RSS. Define decode tokens/s as (N-1)/(last-first), undefined
   for N<2; label differing upstream denominators instead of mixing them.
   Keep prefill throughput and native benchmark metrics alongside these.
   RSS is not GPU memory: report available accelerator memory separately and
   identify missing measurements. Collect profiling in separate matched runs
   so profiling/output overhead does not masquerade as the optimization.
5. Keep ambient temperature, initial thermal state, battery/charging and power
   mode comparable; record device thermal/throttling state throughout. Use a
   validated identical affinity/thread policy, not a copied CPU mask from an
   unrelated phone. Avoid simultaneous benchmarks. Interleave paired OFF/ON
   order with a recorded randomized order/seed; cool to a declared comparable
   state between blocks. Predeclare warmups and enough repeated pairs to
   estimate variability, plus a maximum safe collection window. Report every
   repeat, median, dispersion and paired confidence intervals. If precision
   is inadequate at that boundary, report inconclusive, not a cherry-picked win.

## 5. Evidence and decision

Retain the manifest, commands, logs, output/token traces, raw per-trial timings,
RSS/thermal samples, profiler traces and tool failures with a run ID. Retain
negative findings, slower cases, excluded trials with predefined reasons, and
all crash/error results. Summaries must separate synthetic component timings,
real sampler tests, engine linkage, model inference and actual phone evidence.

STOP the affected run for parity failure, crash/error regression, invalid
configuration, unsafe thermal conditions or an unverified ABI/path. Diagnose
before proceeding. NO-GO for a deployment/default switch if correctness or
platform gates fail. An unexercised optimization or noisy/unchanged timing is
inconclusive performance evidence, not a speedup. GO for considering further
opt-in use only after every target gate passes, exact greedy parity holds, no
crash/error regression is observed in the declared suite, and reproducible
end-to-end benefit exceeds measured uncertainty without a material tradeoff in
TTFT, wall time, memory or sustained thermal behavior. State test coverage;
absence of observed failures is not a universal safety guarantee. Keep the
native default unless separately reviewed and authorized to change it.

Use a measured native time fraction f for the replaced work and its isolated
speedup s: Amdahl's ideal overall speedup is 1/((1-f)+f/s). Copy/readback,
synchronization, executor, tokenization and other costs remain; new overhead
can reduce that bound. Even infinitely fast sampling cannot exceed 1/(1-f).
There is no promised 2x target: decide from target-device end-to-end evidence,
not a component ratio or the implementation language.

### Source map

- [Upstream build/run guide](../docs/getting-started/build-and-run.md)
- [Current simple CLI](../runtime/engine/litert_lm_main.cc),
  [advanced CLI](../runtime/engine/litert_lm_advanced_main.cc),
  [shared flags](../runtime/engine/shared_flags.cc)
- [Session configuration](../runtime/engine/engine_settings.h),
  [metadata/default resolution](../runtime/engine/engine_settings.cc)
- [Actual eligibility/fallback](../runtime/components/top_p_cpu_sampler.cc)
- [Linux engine workflow](../.github/workflows/rust-engine-link.yml),
  [paired real sampler workflow](../.github/workflows/rust-sampler-baseline.yml)
