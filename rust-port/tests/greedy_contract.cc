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
  using rt_rs_experiment::GreedyFp16TokenIds;
  const std::vector<uint16_t> halves = {0x3c00, 0x4200, 0x4200, 0x7c01, 0x4880, 0x3c00};
  auto half_result = GreedyFp16TokenIds(halves, 2, 1);
  REQUIRE(half_result.ok());
  REQUIRE((*half_result)[0] == std::vector<int>({1}));
  REQUIRE((*half_result)[1] == std::vector<int>({0}));
  REQUIRE(!GreedyFp16TokenIds(halves, 0, 1).ok());
  REQUIRE(!GreedyFp16TokenIds(halves, 1, 0).ok());
  REQUIRE(!GreedyFp16TokenIds(halves, -1, 1).ok());
  REQUIRE(!GreedyFp16TokenIds(halves, 2, 2).ok());
  REQUIRE(!GreedyFp16TokenIds(halves, std::numeric_limits<int>::max(), 2).ok());
  auto half_empty = GreedyFp16TokenIds({}, 3, 2);
  REQUIRE(half_empty.ok() && half_empty->size() == 3);
  for (const auto& row : *half_empty) REQUIRE(row == std::vector<int>({0, 0}));
  auto half_owned = [] {
    const std::vector<uint16_t> temporary = {0x8000, 0, 0x4000, 0x4000};
    return GreedyFp16TokenIds(temporary, 1, 1);
  }();
  REQUIRE(half_owned.ok() && (*half_owned)[0][0] == 2);
  // A NaN-seeded lane masks its later maximum in a complete 128-value tile.
  std::vector<uint16_t> tiled(256, 0x3c00);
  tiled[1] = 0x7e00;
  tiled[9] = 0x7c00;
  tiled[130] = 0x4000;
  tiled[131] = 0x4000;
  auto half_tiled = GreedyFp16TokenIds(tiled, 1, 1);
  REQUIRE(half_tiled.ok() && (*half_tiled)[0][0] == 130);
  auto owned = [] {
    const std::vector<float> temporary = {0, 9, 9};
    return GreedyTokenIds(temporary, 1, 1);
  }();
  REQUIRE(owned.ok() && (*owned)[0][0] == 1);
}
