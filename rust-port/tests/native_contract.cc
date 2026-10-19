// Copyright 2026 rt-rs contributors.
// SPDX-License-Identifier: Apache-2.0
// Checks the native public interface, including the actual by-value copy usage.
#ifdef LITERTLM_USE_RUST_STOP_DETECTOR
#include "rust-port/native/stop_token_detector_rust.h"
#else
#include "runtime/components/stop_token_detector.h"
#endif
#include <cstdlib>
#include <iostream>
#include <vector>
#include <limits>
#include <new>
using litert::lm::StopTokenDetector;
#define REQUIRE(c) do { if (!(c)) { std::cerr << "failed: " #c << '\n'; std::abort(); } } while (false)
#ifdef LITERTLM_USE_RUST_STOP_DETECTOR
static size_t fail_after = std::numeric_limits<size_t>::max();
void* operator new(size_t size) {
  if (fail_after != std::numeric_limits<size_t>::max()) {
    if (fail_after == 0) throw std::bad_alloc();
    --fail_after;
  }
  void* result = std::malloc(size == 0 ? 1 : size);
  if (!result) throw std::bad_alloc();
  return result;
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
#endif
int main() {
#ifdef LITERTLM_USE_RUST_STOP_DETECTOR
  for (size_t failure = 0; failure < 7; ++failure) {
    StopTokenDetector detector(2);
    REQUIRE(detector.AddStopTokenSequence({7}).ok());
    fail_after = failure;
    bool caught = false;
    try { detector.ResetBatch(4096); } catch (const std::bad_alloc&) { caught = true; }
    fail_after = std::numeric_limits<size_t>::max();
    REQUIRE(caught);
    REQUIRE(detector.GetStopTokensFound().size() == 2);
    REQUIRE(detector.ProcessTokens(std::vector<std::vector<int>>{{7}, {7}}).ok());
    REQUIRE(detector.AllDone().value());
  }
#endif
#ifdef LITERTLM_USE_RUST_STOP_DETECTOR
  for (size_t failure = 0; failure < 7; ++failure) {
    StopTokenDetector source(2), destination(1);
    REQUIRE(source.AddStopTokenSequence({1, 2}).ok());
    REQUIRE(source.ProcessTokens(std::vector<std::vector<int>>{{1}, {0}}).ok());
    REQUIRE(destination.AddStopTokenSequence({9}).ok());
    fail_after = failure;
    bool caught = false;
    try { destination = source; } catch (const std::bad_alloc&) { caught = true; }
    fail_after = std::numeric_limits<size_t>::max();
    REQUIRE(caught);
    REQUIRE(destination.GetStopTokensFound().size() == 1);
    REQUIRE(destination.ProcessTokens(std::vector<std::vector<int>>{{9}}).ok());
    REQUIRE(destination.AllDone().value());
    REQUIRE(source.MaxPartialStopTokenLength(0) == 1);
  }
#endif
  for (int repeat = 0; repeat < 1000; ++repeat) {
    StopTokenDetector original(2);
    REQUIRE(original.AddStopTokenSequence({1, 2}).ok());
    REQUIRE(original.AddStopTokenSequence({7}).ok());
    REQUIRE(original.ProcessTokens(std::vector<std::vector<int>>{{1}, {7}}).ok());
    StopTokenDetector copy(original);
    REQUIRE(copy.ProcessTokens(std::vector<std::vector<int>>{{2}, {0}}).ok());
    REQUIRE(copy.AllDone().value());
    REQUIRE(!original.AllDone().value());
    REQUIRE(original.GetStepsBeforeStopTokens()[1] == 1);
    REQUIRE(copy.GetStepsBeforeStopTokens()[1] == 2);
    REQUIRE(original.MaxPartialStopTokenLength(0) == 1);
    StopTokenDetector assigned(1);
    assigned = original;
    assigned = assigned;
    REQUIRE(assigned.GetStopTokensFound().size() == 2);
    REQUIRE(assigned.ProcessTokens(std::vector<std::vector<int>>{{2}, {}}).code() == absl::StatusCode::kInternal);
    REQUIRE(assigned.MaxPartialStopTokenLength(0) == 1);
    REQUIRE(assigned.GetStepsBeforeStopTokens()[1] == 1);
    REQUIRE(assigned.ProcessTokens(std::vector<std::vector<int>>{{2}, {0}}).ok());
    REQUIRE(assigned.AllDone().value());
    assigned.ResetBatch(3);
    REQUIRE(assigned.GetStopTokensFound().size() == 3);
    REQUIRE(!assigned.AllDone().value());
    REQUIRE(assigned.ProcessTokens(std::vector<std::vector<int>>{{7}, {7}, {7}}).ok());
    REQUIRE(assigned.AllDone().value());
    REQUIRE(original.GetStopTokensFound().size() == 2);
#ifdef LITERTLM_USE_RUST_STOP_DETECTOR
    REQUIRE(original.MarkForRollback(7));
    original.ResetBatch();
    REQUIRE(original.ProcessTokens(std::vector<std::vector<int>>{{7}, {0}}).ok());
    StopTokenDetector rollback_copy(original);
    REQUIRE(rollback_copy.EndsWithRollbackToken(0));
    REQUIRE(rollback_copy.ProcessTokens(std::vector<std::vector<int>>{{0}, {0}}).ok());
    REQUIRE(!rollback_copy.EndsWithRollbackToken(0));
    REQUIRE(original.EndsWithRollbackToken(0));
#endif
  }
}
