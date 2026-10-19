// Copyright 2026 rt-rs contributors.
// SPDX-License-Identifier: Apache-2.0
// Harness only: greedy sampling uses the unmodified upstream translation unit.
#include <bit>
#include <cstdint>
#include <iostream>
#include <vector>
#include "runtime/components/sampling_cpu_util.h"

int main() {
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
