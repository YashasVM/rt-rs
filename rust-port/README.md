# Incremental Rust port

This fork starts a Rust port of **LiteRT-LM**, the LLM orchestration layer used
by Mobie. The separate [LiteRT repository](https://github.com/google-ai-edge/LiteRT)
implements the lower-level tensor runtime. Google's
[official documentation](https://developers.google.com/edge/litert)
distinguishes these projects.

## Baseline and scope

The reference is upstream LiteRT-LM `v0.18.0`, commit
`b2f686e2ed4718fb84ec398a61dd59ca0f0aff27`. This matches Mobie's pinned
`com.google.ai.edge.litertlm:litertlm-android:0.18.0` dependency at Mobie commit
`d4c6b0381120afead29b0168dd5c132d80af9904`.

`src/lib.rs` translates the flat token-ID `StopTokenDetector` behavior into
Rust. It owns the stop sequences, per-batch progress, completed flags and
discard counts. It contains no FFI and forbids unsafe Rust. The independent
crate has no third-party Cargo dependencies.

The native runtime still uses its original C++ detector. This crate is not yet
selected by the C++ build or connected to Mobie. The complete upstream source,
history, licensing and attribution remain intact. Rust support already present
upstream, such as template and parser components, is not work added by this port.

Nothing in this milestone ports model execution, compute kernels, tokenizers,
sampling, engine/session ownership, conversations, callbacks, cancellation or
device acceleration. Those remain native. There is no inference speedup claim.

## Runtime and build boundaries

The C entry points in `c/engine.cc` and `c/conversation.cc` depend on runtime
conversation, engine, executor and tokenizer libraries. Execution managers in
`runtime/framework/resource_management` create a detector for each session,
retain its configured sequences, and reset its batch for output candidates.
The detector is a separate Bazel target `//runtime/components:stop_token_detector`
and CMake target `LiteRTLM::Runtime::Components::StopTokenDetector`.
The pinned upstream Bazel version is `7.6.1`.

This isolated crate deliberately leaves both native build graphs unchanged.
The reference harness compiles the actual pinned detector `.cc`, not a second
handwritten implementation. Abseil supplies its original status/logging support.
LiteRT headers are required by a transitive status-macro include. They are pinned
separately to `3e49a3789864362f6069e0568435b584a38cf917`; no LiteRT inference
library is linked. The harness creates the CPU-only generated build-config header
inside `target/`, without modifying the LiteRT reference checkout.

## Behavior contract

The implementation preserves these observed upstream behaviors:

- Duplicate sequence additions succeed without changing state.
- New sequences start at progress zero, without replaying earlier tokens.
- A mismatch restarts at length one if the current token equals the first token.
  It does not search for the longest matching suffix. Some overlapping matches
  therefore remain undetected, exactly as in the reference.
- If several sequences complete on the same token, the last one added determines
  the discard count. All sequences still contribute to maximum progress.
- After an item stops, later calls increment its discard count while retaining
  the progress recorded on its stopping step.
- Reset retains sequences and clears per-item state.
- Batch-size and missing-sequence errors do not mutate state.

The Rust API uses `NonZeroUsize` for batch size, `usize` for counters, and `i32`
for token IDs. It returns `Option` for indexed progress. It does not emulate
upstream signed-counter overflow, oversized batch truncation, invalid-index
undefined behavior, constructor aborts or exact error strings. The nested
`vector<vector<int>>` overload is not ported. Parity claims cover defined inputs
to the flat API within the C++ integer range.

## Reproduce verification

Requirements: Rust/Cargo with edition 2021 support, Python 3, a C++20 compiler,
pkg-config and Abseil development headers/libraries with logging support.
No Cargo network access or Python packages are needed.

From the fork root:

```sh
cargo test --manifest-path rust-port/Cargo.toml --locked --offline
cargo clippy --manifest-path rust-port/Cargo.toml --all-targets --offline -- -D warnings
cargo fmt --manifest-path rust-port/Cargo.toml -- --check
```

For native differential tests and microbenchmarks, place the pinned LiteRT
reference beside the fork, or set `LITERT_SOURCE_DIR` to that checkout:

```sh
git clone https://github.com/google-ai-edge/LiteRT.git ../litert-reference
git -C ../litert-reference checkout 3e49a3789864362f6069e0568435b584a38cf917
python3 rust-port/verify.py
```

The verifier rejects changed reference sources and headers. It runs literal
Rust unit tests, exhaustive binary token traces and deterministic random
add/process/reset histories. After every operation it compares status codes,
completion, stopped flags, discard counts and partial progress with C++.
Failures save their input under `target/failed-trace.txt`. Results and environment
metadata are written to ignored `results.json`.

The initial recorded run in `benchmarks/baseline.json` passed 452 traces with
41,135 operations. Its median nanoseconds per four-item batch step were:

| Workload | C++ | Rust |
| --- | ---: | ---: |
| No match | 87.99 | 116.91 |
| Partial match | 79.43 | 109.93 |
| Early stop | 99.61 | 47.54 |

Rust was slower in the first two workloads. The early-stop case mostly measures
already-stopped counters. This first implementation favors parity over tuning.

The benchmark runs both native binaries with optimized builds, two warmups and
seven samples in alternating order. Both execute the same 4-item batches,
three stop sequences and 64-step reset cycles, and must produce equal checksums.
Separate modes exercise nonmatching tokens, partial matches and early stopping.
Reported time includes resets, token preparation and checksum reads. Rust uses
`black_box` around inputs; C++ compiles the detector as a separate translation
unit without LTO. These are local microbenchmarks with synthetic tokens, no
CPU affinity, thermal control, model loading, GPU work or end-to-end inference.
They cannot establish Mobie latency, throughput or a 2x improvement.

## Next integration gate

Before adding an engine bridge, audit `v0.18.0` native callback and cancellation
contracts. A safe wrapper must keep engine/session objects and callback storage
alive until native work has stopped, prevent callbacks from unwinding across FFI,
and prove cleanup, drop/cancel races, ordered chunk delivery and repeated model
inference with the actual native library and model. Mock tests alone cannot prove
those contracts. Keep the original runtime available as a reference and fallback.

## License

This port derives from The ODML Authors' Apache-2.0 source. See the upstream
`LICENSE` retained at the fork root. Translated source retains attribution.
