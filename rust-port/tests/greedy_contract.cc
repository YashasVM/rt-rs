// Copyright 2026 rt-rs contributors.
// SPDX-License-Identifier: Apache-2.0
#include "rust-port/native/greedy_rust.h"
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>
#define REQUIRE(c) do { if (!(c)) { std::cerr << "failed: " #c << '\n'; std::abort(); } } while (false)
int main() {
  using rt_rs_experiment::GreedyTokenIds;
  const std::vector<float> logits = {1, 3, 3, std::numeric_limits<float>::quiet_NaN(), 9, 1};
  for (int repeat = 0; repeat < 1000; ++repeat) {
    auto result = GreedyTokenIds(logits, 2, 1);
    REQUIRE(result.ok());
    REQUIRE((*result)[0][0] == 1);
    REQUIRE((*result)[1][0] == 0);
    REQUIRE(!GreedyTokenIds(logits, 0, 1).ok());
    REQUIRE(!GreedyTokenIds(logits, -1, 1).ok());
    REQUIRE(!GreedyTokenIds(logits, 2, 2).ok());
    REQUIRE(!GreedyTokenIds(logits, std::numeric_limits<int>::max(), 2).ok());
    auto empty = GreedyTokenIds({}, 3, 2);
    REQUIRE(empty.ok() && empty->size() == 3);
    for (const auto& row : *empty) REQUIRE(row == std::vector<int>({0, 0}));
  }
  auto owned = [] {
    const std::vector<float> temporary = {0, 9, 9};
    return GreedyTokenIds(temporary, 1, 1);
  }();
  REQUIRE(owned.ok() && (*owned)[0][0] == 1);
}
