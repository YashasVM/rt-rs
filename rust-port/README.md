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
general top-k/top-p sampling, engine/session ownership, conversations, callbacks, cancellation or
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
`vector<vector<int>>` single-token-per-row overload is ported without an
intermediate flatten allocation. Incorrect row lengths are rejected before
any state changes, after batch-size validation and before missing-sequence
validation, matching the native order. Parity claims cover defined inputs
to the flat API within the C++ integer range.

## Reproduce verification

Requirements: Rust/Cargo 1.70 or newer, Python 3.12 or newer, a C++20 compiler,
pkg-config and Abseil development headers/libraries with logging support.
No Cargo network access or Python packages are needed. If Abseil is installed
outside the system prefix, set `PKG_CONFIG_PATH` to its `lib/pkgconfig` directory;
the verifier obtains its include and static-link flags from pkg-config.

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
The trace adapters consume trusted generated fixtures; malformed trace syntax
can terminate them and is not a public parser/fuzzing contract. API shape errors
are tested separately. Failures save their input under `target/failed-trace.txt`. Results and environment
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

## Greedy logits selection

`src/greedy.rs` adds a second genuine runtime-layer port: the `k = 1` path of
`TopKTokenIds` in the unchanged upstream `sampling_cpu_util.cc`. It is pure Rust,
not an FFI wrapper. `greedy::greedy_token_ids` accepts flattened f32 logits and
nonzero batch/sequence dimensions, returning one vocabulary index per position.
General top-k, softmax, randomized/top-p sampling and runtime integration remain
out of scope.

The 128-element tiles and eight-lane reduction retain the original comparison
order. For inputs without NaNs, ties choose the first index. With NaNs, lane
seeds can mask later elements or whole tiles, so the result can differ from a
simple scalar argmax. The implementation deliberately preserves those reference
behaviors rather than substituting Rust's float `max`. Signed zero and infinities
are covered. Empty vocabulary returns zero indices, matching the native helper.
The Rust API rejects shape multiplication overflow and indivisible buffers,
and uses fallible output allocation. It does not emulate native invalid zero
dimensions, signed overflow, or exact error strings.

Run `python3 rust-port/verify_greedy.py` for the unchanged C++ translation-unit
comparison. This verifier needs Abseil but not the separate LiteRT checkout.
The cloud run passed 11,481 deterministic cases: every maximum/tie/NaN position
across tile-boundary vocabulary sizes, special IEEE-754 bit patterns, randomized
float bit patterns, batch/sequence layouts, empty vocabularies and invalid buffer
shapes. Nineteen greedy unit tests plus the five stop-detector tests also pass.
The recorded environment and input digest are in `benchmarks/cloud-greedy.json`.
The same cases also passed an AddressSanitizer/UndefinedBehaviorSanitizer native
harness run (`benchmarks/cloud-greedy-sanitized.json`). Leak checking was disabled
because LeakSanitizer cannot run under this executor's ptrace; the separately
built Abseil libraries were not instrumented. No greedy performance or
end-to-end inference improvement is claimed.

## Cloud stop-detector follow-up

The reset/iteration optimization at checkpoint `230f07f` reuses allocated state
and avoids repeated indexed batch access. Cloud verification again passed all
452 C++ traces / 41,135 operations. `benchmarks/cloud-stop.json` records a fresh
C++/Rust comparison; `benchmarks/cloud-stop-ab.json` compares the original Rust
milestone against this optimized implementation using the same compiler/machine.
The latter medians fell by 17.9% (no match), 22.8% (partial match) and 7.4%
(early stop). Those are component microbenchmarks, not model inference.
Run `python3 rust-port/compare.py` to reproduce the A/B check. This command
requires a Cargo version supporting the original milestone's v4 lockfile
(tested with Cargo 1.99.1); the Rust 1.70 minimum applies to the current crate,
not that archived comparison build. Run it serially,
not alongside the differential verifier or another benchmark/build.

These samples come from a different CPU/compiler environment than the original
baseline and must not be compared across machines. Both recorded cloud checks
were run serially after this port's dependency build. Unrelated cloud jobs and
shared-host scheduling were not controlled, so timing comparisons are provisional.
The lack of CPU affinity/thermal control still applies.
The current source digest in the A/B report identifies the working-tree library;
its commit field records the parent checkout before the new port commit.

## Pinned current-main compatibility

The current-main comparison is pinned to upstream LiteRT-LM commit
`e32526ccbc54fed8d7ba8fbf292c03572fba5081`, observed on 2026-10-19. It is not a
claim of compatibility with future moving main. Fetch the exact object, then run:

```sh
git fetch https://github.com/google-ai-edge/LiteRT-LM.git e32526ccbc54fed8d7ba8fbf292c03572fba5081
RT_RS_CURRENT_MAIN=1 python3 rust-port/verify.py
RT_RS_CURRENT_MAIN=1 python3 rust-port/verify_greedy.py
```

The current-main stop detector adds singleton stop-token rollback marking.
`mark_for_rollback` and `ends_with_rollback_token` now preserve its behavior:
marks survive reset, per-item flags clear on reset or the next processed token,
marking is not retroactive, duplicate additions retain marks, and the last
simultaneous match controls the flag. Invalid input leaves all state untouched.
This records the rollback decision only; it does not manipulate an executor or
KV cache. The current-main greedy source/header are byte-identical to v0.18.0;
the main-mode verifier checks that identity before testing.

The main detector verifier materializes the exact pinned source/header and
status-macro files under target/, without editing upstream tracked code. It
compares rollback flags in addition to all previously observed state. Both
nested and flat synthetic microbenchmarks run serially with equal checksums.
The expanded compatibility run passes 582 traces / 60,736 operations at
v0.18.0 and 713 traces / 86,492 operations at pinned main, plus 30 Rust unit
tests. Reports are `benchmarks/cloud-stop-nested.json`,
`benchmarks/cloud-stop-main.json`, and `benchmarks/cloud-greedy-main.json`.
Use `RT_RS_SKIP_BENCHMARKS=1` for parity-only detector verification.
Nested benchmarks preallocate caller row containers in both implementations;
the C++ detector still flattens to a temporary vector per call, while Rust
processes validated rows directly. No allocator instrumentation or whole-runtime
speed claim is implied by this source-level allocation removal.

## Continuous integration

The Rust-only workflow runs offline locked unit tests, fmt and clippy on pinned
Rust 1.70.0, with checkout pinned to a verified commit and read-only repository
permissions. Native differential checks are separately reproducible with the
commands above; they are not part of this lightweight workflow. Workflow success
does not establish model inference correctness or Android device performance.

## Representative greedy profiling

`benchmark_greedy.py` separates build/parity work from timing:

```sh
python3 rust-port/benchmark_greedy.py --phase build
python3 rust-port/benchmark_greedy.py --phase run
```

It uses vocabulary sizes 32,768 / 131,072 / 262,144, batch 1 / 4, and sequence
1 / 8. These are representative shapes, not a tested model or device. It rotates
candidate ordering, records five samples after two warmups, and compares index
checksums. Finite deterministic logits are used for timings; the separate
11,481-case parity suite retains NaN/tie edge coverage.

The first profile (`benchmarks/cloud-greedy-profile.json`) finds the Rust
reduction 1.29–3.14 times faster than GCC C++ reduction across these twelve
shapes. Examples: 32,768 vocabulary, batch/sequence 1/1, 6.00 to 1.91 microseconds
per position; 262,144 vocabulary, 1/1, 41.62 to 17.04 microseconds. This compares
GCC 14 with Rust's LLVM, not matched compiler backends. It is not proof of a
language-intrinsic, Android, integrated runtime or inference improvement.

The native full greedy sampling path additionally flattens top IDs, computes
softmax and allocates temporary probabilities/maxima before returning scores of
one. A separately instrumented allocation diagnostic counts 16 ordinary C++
`new` calls / 164 requested bytes for its first batch/sequence 1/1 call, versus
3 / 32 for reduction alone, excluding preallocated logits/RNG. Instrumentation
is absent from timing binaries and is not a complete allocator profiler.
Full-sampling versus reduction timings are close/noisy relative to the large
vocabulary scan: allocation removal alone is not established as the primary
bottleneck. Next test whether the measured reduction advantage survives a real
native boundary before broadening or selecting a Rust backend.

### Stateless greedy boundary experiment

The same benchmark now includes `native_rust_reduction`: a distinct C++
`rt_rs_experiment::GreedyTokenIds` wrapper calling the safe reduction through
`rt_rs_greedy_token_ids`. It borrows immutable logits synchronously, validates
shape/index ranges and buffer metadata before writes, and retains no pointers.
The Rust reduction allocates nothing; the measured C++ wrapper deliberately
includes a flat export buffer and copy into its owned nested result. Inputs and
outputs must be disjoint, live and aligned; no concurrent mutation of inputs or
concurrent access to the output is permitted. Unexpected panics are caught, but
output rollback on an unexpected mid-write panic is not promised.

Including the native boundary, the periodic profile still showed a 1.15–3.12x
median advantage over GCC native reduction (`cloud-native-greedy-profile.json`).
Separate seeded-random, ascending and descending profiles cover another 36
shape/distribution combinations. Median ratios were 1.055–3.31x, 1.30–1.46x,
and 1.11–3.51x respectively, with matching index checksums. The weakest 5–11%
advantages are particularly vulnerable to shared-host noise. Do not summarize
this as a universal threefold gain. All raw samples are in the corresponding
`benchmarks/cloud-native-greedy-{random,ascending,descending}.json` files.

Reproduce a sensitivity run after building with `--pattern random`,
`--pattern ascending`, or `--pattern descending` on the benchmark script.
Compiler, CPU, synthetic-logit and lack-of-inference limitations still apply.

This remains a standalone experiment with no original sampling target/default
change. The 11,481-case suite checks both the safe Rust executable and this
native boundary against unchanged pinned C++ including NaNs, infinities, ties,
empty vocabulary and tiled reductions. Native contract tests also check invalid
shapes and output ownership. Run it directly with:

```sh
RT_RS_CURRENT_MAIN=1 RT_RS_GREEDY_FFI=1 python3 rust-port/verify_greedy.py
```

## Standalone native-adapter experiment

`ffi/` isolates the unsafe C ABI from the safe core. `native/` implements a
separate C++ `StopTokenDetector` adapter for contract and boundary-cost tests.
This is **incomplete integration**, not a selectable production backend. The
original engine headers, CMake/Bazel graphs and default implementation are
unchanged. Do not link the experimental class and the original class into the
same binary: their identical C++ names have different layouts.

On native Linux x86-64 GNU, with Cargo, CMake and the documented Abseil pkg-config
metadata available:

```sh
cmake -S rust-port/native -B rust-port/target/native-cmake
cmake --build rust-port/target/native-cmake -j2
ctest --test-dir rust-port/target/native-cmake --output-on-failure
RT_RS_NATIVE_BACKEND=1 RT_RS_SKIP_BENCHMARKS=1 python3 rust-port/verify.py
RT_RS_NATIVE_BACKEND=1 RT_RS_CURRENT_MAIN=1 RT_RS_SKIP_BENCHMARKS=1 python3 rust-port/verify.py
```

The adapter owns one opaque Rust handle. Copy construction/assignment deeply
clone state; destruction releases it exactly once. Calls are synchronous, with
no callbacks or cancellation work. Callers must serialize access and obey the
pointer/alignment/lifetime/non-aliasing contract in `native/stop_ffi.h`. Metadata
checks cannot make arbitrary dangling pointers safe. Rust unwinds are caught and
poison the handle; allocation aborts, signals and `panic=abort` overrides are not
recoverable. C++ allocation exceptions are allowed; mirror resizing commits only
after all allocations succeed. Contract tests inject failure at each of seven
resize allocations and seven copy-assignment allocations, and exercise copies,
assignment and invalid-row atomicity.

Nested native inputs are copied into retained Rust scratch storage, and each
successful step copies result snapshots into C++ mirrors. This avoids temporary
per-step allocations after sizing but is not a zero-copy interface. These costs
must be measured rather than assumed faster. A cloud native contract run also passes AddressSanitizer/UndefinedBehaviorSanitizer
on the C++ adapter and harness; the Rust archive and Abseil are uninstrumented
and leak detection was disabled. This is not whole-program sanitizer coverage.
Rust-only CI includes the seven ABI
unit tests, not native C++ parity, sanitizer or full-engine integration tests.

The first measured adapter boundary is slower, not faster. Against pinned main,
median native-to-Rust cost increased by 70.5% / 31.3% / 24.8% for flat no-match /
partial / early-stop, and 35.1% / 22.4% / 24.2% for the corresponding nested
modes. Raw samples, source/binary hashes and caveats are recorded in
`benchmarks/cloud-native-stop-main.json`. Other known project jobs paused during
this short run, but CPU frequency, affinity and external host load were not
controlled. These component results reject assuming that pure-Rust helper gains
survive a fine-grained C++ boundary. Reduce/coarsen copying and snapshot work
before considering a real backend switch; never infer overall runtime speed.

A bounded follow-up fuses processing and snapshot export under one Rust handle
borrow and panic guard. It keeps eager C++ mirrors, including references obtained
before processing; inputs may alias output storage because all input borrows end
before snapshot writes. Regression tests cover using a getter as input and
invalid fused metadata without state/output changes. The fused variant is still
slower than C++ by 61.3% / 19.6% / 20.9% (flat) and 31.3% / 17.8% / 31.2%
(nested) in its run; see `benchmarks/cloud-native-fused-stop-main.json`. Different
runs are not controlled A/B evidence of a fusion speedup. Further tiny-detector
tuning is deferred in favor of realistic-vocabulary sampling profiling.

Actual engine integration still requires a single consistent class layout for
all consumers (including serial/threaded execution managers), inclusion of the
Rust archive in the final flattened archive registry, and builds/tests of those
actual targets. Standalone link success cannot establish either requirement.
No Android, model inference, device performance or end-to-end gain is validated.

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
