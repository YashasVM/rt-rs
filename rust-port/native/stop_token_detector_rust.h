// Copyright 2025 The ODML Authors.
// Copyright 2026 rt-rs contributors.
// SPDX-License-Identifier: Apache-2.0
// Standalone experiment implementing the native StopTokenDetector interface.
#ifndef RT_RS_NATIVE_STOP_TOKEN_DETECTOR_H_
#define RT_RS_NATIVE_STOP_TOKEN_DETECTOR_H_
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "rust-port/native/stop_ffi.h"

namespace litert::lm {
// Synchronous and not thread-safe, matching the reference. No call retains input
// pointers or schedules work. Deep copies include in-flight matching state.
// Custom copy/destruction intentionally suppress implicit pointer-moving moves.
// Never link this class with the original detector in the same binary.
// This header is not selected by the production engine build.
class StopTokenDetector {
 public:
  explicit StopTokenDetector(size_t batch_size);
  ~StopTokenDetector();
  StopTokenDetector(const StopTokenDetector&);
  StopTokenDetector& operator=(const StopTokenDetector&);
  absl::Status AddStopTokenSequence(const std::vector<int>& stop_sequence);
  void ResetBatch(size_t batch_size = 0);
  absl::Status ProcessTokens(absl::Span<const int> tokens);
  absl::Status ProcessTokens(const std::vector<std::vector<int>>& tokens);
  const std::vector<int>& GetStepsBeforeStopTokens() const;
  absl::StatusOr<bool> AllDone() const;
  const std::vector<bool>& GetStopTokensFound() const { return found_; }
  int MaxPartialStopTokenLength(int index) const;
  bool MarkForRollback(int token);
  bool EndsWithRollbackToken(int index) const;
 private:
  struct Deleter { void operator()(RtRsStopDetector* p) const { rt_rs_stop_destroy(p); } };
  std::unique_ptr<RtRsStopDetector, Deleter> detector_;
  std::vector<bool> found_;
  std::vector<uint8_t> flags_, rollback_;
  std::vector<int> discard_, partial_;
  std::vector<const int32_t*> row_pointers_;
  std::vector<size_t> row_lengths_;
  void Resize(size_t size);
  absl::Status Refresh();
  void Swap(StopTokenDetector& other) noexcept;
};
}
#endif
