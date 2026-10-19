// Copyright 2025 The ODML Authors.
// Copyright 2026 rt-rs contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Experimental C++ control for rust-port/src/half.rs, not upstream production
// code. Uses the same branchless integer-key formulation and eight-lane,
// 128-element reduction to distinguish algorithm gains from language/compiler
// effects. No float conversion, allocation, SIMD intrinsics, or fast-math
// assumptions are required. "Branchless" describes the key/mask formulation,
// not a guarantee about machine code emitted by any particular compiler.

#ifndef RT_RS_TESTS_HALF_FUSED_CONTROL_H_
#define RT_RS_TESTS_HALF_FUSED_CONTROL_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "absl/types/span.h"

namespace rt_rs_experiment {
namespace half_fused_control_internal {

inline bool IsNan(uint16_t bits) {
  return (bits & 0x7fffu) > 0x7c00u;
}

// Explicit narrowing reproduces u16 wrapping subtraction and complement,
// independent of C++ integer promotions. All subtraction is unsigned.
inline uint16_t Mask(bool condition) {
  return static_cast<uint16_t>(0u - static_cast<unsigned>(condition));
}

inline uint16_t OrderKey(uint16_t bits) {
  const uint16_t sign_mask =
      static_cast<uint16_t>(0u - (static_cast<unsigned>(bits) >> 15));
  const uint16_t key = static_cast<uint16_t>(bits ^ (sign_mask | 0x8000u));
  const uint16_t zero_mask = Mask((bits & 0x7fffu) == 0);
  const uint16_t nonzero_mask = static_cast<uint16_t>(~zero_mask);
  // Both signed zeros map to 0x8000, preserving earliest-index zero ties.
  return static_cast<uint16_t>((key & nonzero_mask) | (0x8000u & zero_mask));
}

// Precondition: tile points to 128 readable binary16 values. Returning false
// corresponds to Rust's None: NaN-seeded lane zero masks the entire tile.
inline bool TileMaxKey(const uint16_t* tile, uint16_t& maximum) {
  if (IsNan(tile[0])) return false;
  uint16_t keys[8] = {};
  bool masked[8] = {};
  for (size_t lane = 0; lane < 8; ++lane) {
    masked[lane] = IsNan(tile[lane]);
    keys[lane] = static_cast<uint16_t>(OrderKey(tile[lane]) & Mask(!masked[lane]));
  }
  for (size_t group = 8; group < 128; group += 8) {
    for (size_t lane = 0; lane < 8; ++lane) {
      const uint16_t bits = tile[group + lane];
      const uint16_t valid_mask = Mask(!masked[lane] && !IsNan(bits));
      keys[lane] = std::max(keys[lane],
                           static_cast<uint16_t>(OrderKey(bits) & valid_mask));
    }
  }
  // Valid numbers, including -infinity, have strictly positive keys. Zero is
  // therefore a safe sentinel for permanently NaN-seeded lanes.
  maximum = *std::max_element(keys, keys + 8);
  return true;
}

}  // namespace half_fused_control_internal

// Exact scalar API counterpart of greedy_fp16_token_id: empty input or a NaN
// first element returns zero. Full tiles retain the native reduction's masked
// lanes; the tail uses ordinary sequential comparisons. Only a strict increase
// replaces the index, so ties retain the first match. Equality search includes
// every tile element, even those in masked lanes, just as in the Rust version.
inline size_t NativeFusedHalfTokenId(absl::Span<const uint16_t> logits) {
  using half_fused_control_internal::IsNan;
  using half_fused_control_internal::OrderKey;
  using half_fused_control_internal::TileMaxKey;
  if (logits.empty() || IsNan(logits[0])) return 0;
  uint16_t maximum = OrderKey(logits[0]);
  size_t maximum_index = 0;
  size_t offset = 0;
  // Subtraction avoids overflow in a hypothetical near-SIZE_MAX span.
  for (; logits.size() - offset >= 128; offset += 128) {
    const uint16_t* tile = logits.data() + offset;
    uint16_t candidate;
    if (TileMaxKey(tile, candidate) && candidate > maximum) {
      for (size_t index = 0; index < 128; ++index) {
        const uint16_t bits = tile[index];
        if (!IsNan(bits) && OrderKey(bits) == candidate) {
          maximum = candidate;
          maximum_index = offset + index;
          break;
        }
      }
    }
  }
  for (; offset < logits.size(); ++offset) {
    const uint16_t bits = logits[offset];
    if (!IsNan(bits) && OrderKey(bits) > maximum) {
      maximum = OrderKey(bits);
      maximum_index = offset;
    }
  }
  return maximum_index;
}

}  // namespace rt_rs_experiment

#endif  // RT_RS_TESTS_HALF_FUSED_CONTROL_H_
