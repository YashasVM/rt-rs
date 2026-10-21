# Genuine-model CPU diagnostic (not a benchmark)

This Linux-only, test-only binary loads a real `.litertlm` asset through the
normal EngineFactory and Conversation APIs. It does not change production
sources, manually emulate the tokenizer/template, or replace native sampling.
It requests CPU execution and CPU sampling, two CPU threads, TOP_P with k=1,
p=1, temperature=1, seed=0, speculative decoding disabled, and at most 16 output
tokens. The fixed prompt is recorded in its JSON report.

## Observation mechanism

Public Engine/Session settings do not offer an injectable Sampler factory.
The diagnostic therefore uses GNU-compatible `--wrap` on the cross-translation-
unit CreateSampler call, invokes the real factory exactly once, and decorates
its returned Sampler. The decorator forwards all Sampler methods and samples
once through the original implementation. It records actual logits dtype and
shape, whether scores were requested, sampler status, sampled token IDs, and
factory/update configuration. Extra tensor reads and logging make this target
unsuitable for performance comparisons.

The wrapper's signature is checked against the real header at compile time.
Its Itanium ABI symbol is explicitly pinned in the test BUILD and source.
Static linking is required: a wrapper symbol being present is not proof that it
ran. Both the binary and validator require positive factory and sampling call
counts. Output IDs include sampled stop/EOS IDs if encountered; they need not
all appear in the rendered assistant text. The response JSON contains the text
that the actual Conversation produced.

The report also includes resolved engine metadata/settings, resolved CPU
thread count, Conversation session configuration, selected Jinja template,
output cap, and final status. Template absence or incompatible model metadata
fails normally. Unknown dtype, unexpected ID layout, more than the bounded
number of observations, or absent instrumentation fails rather than silently
claiming success. Errors in observation itself are diagnostic failures, not
claims about production behavior.

## Bounded hosted run

`.github/workflows/rust-model-probe.yml` is a single standard ubuntu-24.04 job,
60-minute outer timeout, Bazel jobs=2 and RAM budget 6144 MB, CPU-only build,
Rust FP16 opt-in explicitly false. It checks out the immutable triggering SHA
without persisted credentials and repairs only the same pinned upstream LFS
constraint provider used by the earlier engine-link experiment. The real
Conversation run has a separate 180-second timeout and 10-second kill grace.
There are no paid runners, secrets, remote caches, new dependencies, or
production deployments. Build errors are not suppressed.

After building, it downloads only this approved public Apache-2.0 model:

- Repository: litert-community/SmolLM2-135M-Instruct
- Revision: 749fc23bc8d7ef89db031d075004289c88f41c2f
- File: SmolLM2_135M_Instruct.litertlm
- Size: 142819328 bytes
- SHA256: ccdc5c85735743f081b7d44ca309cab569f76c0f2f0e8e163449a63721969c37

No gated authentication, new agreement, or login flow is attempted. Both model
and dependency downloads require exact size and SHA256 verification. Reports
are printed to job logs; model files are not committed or uploaded as artifacts.

## Interpretation and remaining validation

The first run is **fork native OFF**, not untouched upstream main and not a
Rust speed result. Source inspection cannot establish model compatibility,
actual CPU logits dtype, generated IDs/text, or successful factory interception.
Only a successful genuine hosted inference can establish these observations.
FP32 logits would not exercise the existing Rust FP16-only opt-in. A later
matched native/Rust comparison must first choose an actually exercised path,
preserve identical source/model/settings, verify invocation and outputs, and
use an uninstrumented timing harness before making any performance claim.

Local checks cover workflow YAML/shell syntax and synthetic validator cases;
no full engine build or model inference is performed locally.
