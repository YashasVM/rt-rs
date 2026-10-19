// Copyright 2026 rt-rs contributors.
// SPDX-License-Identifier: Apache-2.0
#include "rust-port/native/greedy_rust.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>
#include "absl/status/status.h"
#include "rust-port/native/stop_ffi.h"
static_assert(std::is_same_v<int, int32_t>);
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
namespace rt_rs_experiment {
absl::StatusOr<std::vector<std::vector<int>>> GreedyTokenIds(
    absl::Span<const float> logits, int batch, int sequence) {
  if (batch <= 0 || sequence <= 0) return absl::InvalidArgumentError("Positive shape required");
  const uint64_t positions = static_cast<uint64_t>(batch) * sequence;
  if (positions > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
      logits.size() > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      logits.size() % positions != 0) return absl::InvalidArgumentError("Invalid native greedy shape");
  // Explicitly include flatten/export/copy allocation overhead in measurements.
  std::vector<int> flat(static_cast<size_t>(positions));
  const int status = rt_rs_greedy_token_ids(logits.data(), logits.size(), batch, sequence,
                                           flat.data(), flat.size());
  if (status != 0) return absl::Status(static_cast<absl::StatusCode>(status), "Rust greedy failure");
  std::vector<std::vector<int>> output(batch, std::vector<int>(sequence));
  for (int b = 0; b < batch; ++b) {
    std::copy_n(flat.begin() + static_cast<size_t>(b) * sequence, sequence, output[b].begin());
  }
  return output;
}
absl::StatusOr<std::vector<std::vector<int>>> GreedyFp16TokenIds(
    absl::Span<const uint16_t> logits, int batch, int sequence) {
  if (batch <= 0 || sequence <= 0) return absl::InvalidArgumentError("Positive shape required");
  const uint64_t positions = static_cast<uint64_t>(batch) * sequence;
  if (positions > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
      logits.size() > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      logits.size() % positions != 0) return absl::InvalidArgumentError("Invalid native greedy shape");
  // Explicitly include flatten/export/copy allocation overhead in measurements.
  std::vector<int> flat(static_cast<size_t>(positions));
  const int status = rt_rs_greedy_fp16_token_ids(logits.data(), logits.size(), batch, sequence,
                                           flat.data(), flat.size());
  if (status != 0) return absl::Status(static_cast<absl::StatusCode>(status), "Rust greedy failure");
  std::vector<std::vector<int>> output(batch, std::vector<int>(sequence));
  for (int b = 0; b < batch; ++b) {
    std::copy_n(flat.begin() + static_cast<size_t>(b) * sequence, sequence, output[b].begin());
  }
  return output;
}
}
