// Copyright 2026 rt-rs contributors.
// SPDX-License-Identifier: Apache-2.0
// Harness only: greedy sampling uses the unmodified upstream translation unit.
#include <bit>
#include <chrono>
#include <cstdlib>
#include <new>
#include <memory>
#include <random>
#include <string>
#include <cstdint>
#include <iostream>
#include <vector>
#include "runtime/components/sampling_cpu_util.h"

#ifdef RT_RS_COUNT_ALLOCATIONS
// Separate diagnostic build only; never use these instrumented timings.
static size_t allocations = 0, allocated_bytes = 0;
void* operator new(size_t size) {
  void* p = std::malloc(size ? size : 1);
  if (!p) throw std::bad_alloc();
  ++allocations;
  allocated_bytes += size;
  return p;
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
#endif

int main(int argc, char** argv) {
  if (argc == 6) {
    const bool sampling = std::string(argv[1]) == "sampling";
    const int batch = std::stoi(argv[2]), sequence = std::stoi(argv[3]);
    const int vocab = std::stoi(argv[4]), rounds = std::stoi(argv[5]);
    if (batch <= 0 || sequence <= 0 || vocab <= 0 || rounds <= 0) return 2;
    std::vector<float> logits(static_cast<size_t>(batch) * sequence * vocab);
    for (size_t i = 0; i < logits.size(); ++i) logits[i] = static_cast<int>(i % 101) - 50;
    auto rng = std::make_shared<std::default_random_engine>(1800);
    const auto initial_rng = *rng;
    std::vector<std::vector<float>> scores;
    uint64_t checksum = 0;
#ifdef RT_RS_COUNT_ALLOCATIONS
    allocations = allocated_bytes = 0;
#endif
    const auto start = std::chrono::steady_clock::now();
    for (int repeat = 0; repeat < rounds; ++repeat) {
      logits[0] = repeat % 7 - 3;
      auto ids = sampling
          ? litert::lm::TopKTopPSampling(logits, 1, 0.9f, 1.0f, rng, batch, sequence, scores)
          : litert::lm::TopKTokenIds(logits, 1, batch, sequence);
      if (!ids.ok()) return 3;
      for (const auto& row : *ids) for (int id : row) checksum += id;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count();
    if (*rng != initial_rng) return 4;
    if (sampling) for (const auto& row : scores) for (float score : row) if (score != 1.0f) return 5;
#ifdef RT_RS_COUNT_ALLOCATIONS
    const size_t count = allocations, bytes = allocated_bytes;
    std::cout << count << ' ' << bytes << ' ' << checksum << '\n';
#else
    std::cout << elapsed << ' ' << checksum << '\n';
#endif
    return 0;
  }

  int batch, sequence;
  size_t count;
  while (std::cin >> batch >> sequence >> count) {
    std::vector<float> logits(count);
    // Keep an allocated element for empty input to avoid null pointer arithmetic
    // in the upstream wrapper, while the supplied span remains empty.
    logits.reserve(count + 1);
    for (auto& value : logits) {
      uint32_t bits;
      if (!(std::cin >> std::hex >> bits >> std::dec)) return 2;
      value = std::bit_cast<float>(bits);
    }
    auto result = litert::lm::TopKTokenIds(logits, 1, batch, sequence);
    if (!result.ok()) {
      std::cout << "ERR\n";
      continue;
    }
    std::cout << "OK";
    for (const auto& row : *result) for (int token : row) std::cout << ' ' << token;
    std::cout << '\n';
  }
}
