// Copyright 2026 rt-rs contributors.
// SPDX-License-Identifier: Apache-2.0
#ifndef RT_RS_EXPERIMENT_GREEDY_RUST_H_
#define RT_RS_EXPERIMENT_GREEDY_RUST_H_
#include <cstdint>
#include <vector>
#include "absl/status/statusor.h"
#include "absl/types/span.h"
namespace rt_rs_experiment {
// Standalone experiment, not selected by the production sampling target.
// Borrows logits only for this synchronous call; owns the returned nested rows.
// Allocation exceptions propagate. No RNG state, callbacks or retained pointers.
absl::StatusOr<std::vector<std::vector<int>>> GreedyTokenIds(
    absl::Span<const float> logits, int batch, int sequence);
// Binary16 encodings; same ownership contract. Includes flat export allocation
// and copy into owned nested indices, but no float32 logits staging.
absl::StatusOr<std::vector<std::vector<int>>> GreedyFp16TokenIds(
    absl::Span<const uint16_t> logits, int batch, int sequence);
}
#endif
