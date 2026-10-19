// Copyright 2025 The ODML Authors.
// Copyright 2026 rt-rs contributors.
// SPDX-License-Identifier: Apache-2.0
#include "rust-port/native/stop_token_detector_rust.h"
#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include "absl/log/absl_check.h"

#ifndef LITERTLM_USE_RUST_STOP_DETECTOR
#error "This source requires the opt-in Rust backend macro on every consumer."
#endif
static_assert(std::is_same_v<int, int32_t>, "Rust stop ABI requires 32-bit int");
namespace litert::lm {
namespace {
absl::Status Status(int32_t code) {
  return code == 0 ? absl::OkStatus() : absl::Status(static_cast<absl::StatusCode>(code), "Rust stop detector error");
}
RtRsStopDetector* Create(size_t batch) {
  ABSL_CHECK_GT(batch, 0);
  ABSL_CHECK_LE(batch, static_cast<size_t>(std::numeric_limits<int>::max()));
  auto* result = rt_rs_stop_create(batch);
  if (!result) throw std::bad_alloc();
  return result;
}
}
StopTokenDetector::StopTokenDetector(size_t batch) : detector_(Create(batch)) {
  Resize(batch);
  ABSL_CHECK_OK(Refresh());
}
StopTokenDetector::~StopTokenDetector() = default;
StopTokenDetector::StopTokenDetector(const StopTokenDetector& other)
    : detector_(rt_rs_stop_clone(other.detector_.get())), found_(other.found_),
      flags_(other.flags_), rollback_(other.rollback_), discard_(other.discard_),
      partial_(other.partial_), row_pointers_(other.found_.size()),
      row_lengths_(other.found_.size(), 1) {
  if (!detector_) throw std::runtime_error("Unable to clone Rust stop detector");
}
StopTokenDetector& StopTokenDetector::operator=(const StopTokenDetector& other) {
  if (this != &other) { StopTokenDetector copy(other); Swap(copy); }
  return *this;
}
void StopTokenDetector::Swap(StopTokenDetector& other) noexcept {
  detector_.swap(other.detector_); found_.swap(other.found_);
  flags_.swap(other.flags_); rollback_.swap(other.rollback_);
  discard_.swap(other.discard_); partial_.swap(other.partial_);
  row_pointers_.swap(other.row_pointers_); row_lengths_.swap(other.row_lengths_);
}
void StopTokenDetector::Resize(size_t size) {
  if (found_.size() == size) return;
  // Strong exception guarantee: do not expose a new batch size until every
  // mirror and row-metadata allocation has succeeded.
  std::vector<bool> found(size);
  std::vector<uint8_t> flags(size), rollback(size);
  std::vector<int> discard(size), partial(size);
  std::vector<const int32_t*> pointers(size);
  std::vector<size_t> lengths(size, 1);
  found_.swap(found); flags_.swap(flags); rollback_.swap(rollback);
  discard_.swap(discard); partial_.swap(partial);
  row_pointers_.swap(pointers); row_lengths_.swap(lengths);
}
absl::Status StopTokenDetector::Refresh() {
  auto status = Status(rt_rs_stop_snapshot(detector_.get(), flags_.data(), discard_.data(),
                                          partial_.data(), rollback_.data(), found_.size()));
  if (status.ok()) for (size_t i = 0; i < found_.size(); ++i) found_[i] = flags_[i] != 0;
  return status;
}
absl::Status StopTokenDetector::AddStopTokenSequence(const std::vector<int>& sequence) {
  return Status(rt_rs_stop_add(detector_.get(), sequence.data(), sequence.size()));
}
void StopTokenDetector::ResetBatch(size_t batch) {
  const size_t size = batch == 0 ? found_.size() : batch;
  ABSL_CHECK_LE(size, static_cast<size_t>(std::numeric_limits<int>::max()));
  Resize(size);
  ABSL_CHECK_EQ(rt_rs_stop_reset(detector_.get(), batch), 0);
  ABSL_CHECK_OK(Refresh());
}
absl::Status StopTokenDetector::ProcessTokens(absl::Span<const int> tokens) {
  auto status = Status(rt_rs_stop_process(detector_.get(), tokens.data(), tokens.size()));
  return status.ok() ? Refresh() : status;
}
absl::Status StopTokenDetector::ProcessTokens(const std::vector<std::vector<int>>& tokens) {
  if (tokens.size() != found_.size()) return absl::InvalidArgumentError("Batch size mismatch");
  for (const auto& row : tokens) if (row.size() != 1) return absl::InternalError("Expected singleton row");
  for (size_t i = 0; i < tokens.size(); ++i) row_pointers_[i] = tokens[i].data();
  auto status = Status(rt_rs_stop_process_rows(detector_.get(), row_pointers_.data(), row_lengths_.data(), tokens.size()));
  return status.ok() ? Refresh() : status;
}
const std::vector<int>& StopTokenDetector::GetStepsBeforeStopTokens() const { return discard_; }
absl::StatusOr<bool> StopTokenDetector::AllDone() const {
  return std::all_of(found_.begin(), found_.end(), [](bool value) { return value; });
}
int StopTokenDetector::MaxPartialStopTokenLength(int index) const { return partial_.at(index); }
bool StopTokenDetector::MarkForRollback(int token) { return rt_rs_stop_mark(detector_.get(), token) == 0; }
bool StopTokenDetector::EndsWithRollbackToken(int index) const { return rollback_.at(index) != 0; }
}
