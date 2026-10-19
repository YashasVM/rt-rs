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
// Host-only reference harness. ConvertFp16ToFp32 below is copied verbatim from
// runtime/components/top_p_cpu_sampler.cc in LiteRT-LM v0.18.0
// (b2f686e2ed4718fb84ec398a61dd59ca0f0aff27) and is identical in upstream main
// e32526ccbc54fed8d7ba8fbf292c03572fba5081. Greedy sampling links the unchanged
// upstream runtime/components/sampling_cpu_util.cc translation unit.
// tflite::half comes from the actual LiteRT header, not a replacement:
// LiteRT 3e49a3789864362f6069e0568435b584a38cf917, tflite/types/half.h.
// Conversion behavior is specific to the compiler, flags, and chosen header
// path; this harness alone does not validate an engine, model, or device.

#include <bit>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "runtime/components/sampling_cpu_util.h"
#include "tflite/types/half.h"

namespace {

// Converts an array of float16 values to float32 values.
void ConvertFp16ToFp32(absl::Span<const uint16_t> fp16_values,
                       std::vector<float>& out) {
  out.resize(fp16_values.size());
  for (int i = 0; i < fp16_values.size(); ++i) {
    tflite::half half_val;
    std::memcpy(&half_val, fp16_values.data() + i, sizeof(uint16_t));
    out[i] = static_cast<float>(half_val);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "convert") {
    std::vector<uint16_t> fp16_values(65536);
    for (size_t i = 0; i < fp16_values.size(); ++i) {
      fp16_values[i] = static_cast<uint16_t>(i);
    }
    std::vector<float> converted;
    ConvertFp16ToFp32(fp16_values, converted);
    for (float value : converted) {
      std::cout << std::hex << std::setw(8) << std::setfill('0')
                << std::bit_cast<uint32_t>(value) << '\n';
    }
    return 0;
  }
  if (argc == 7) {
    const std::string mode = argv[1], pattern = argv[6];
    if (mode != "staged" && mode != "staged-copy") return 2;
    const int batch = std::stoi(argv[2]), sequence = std::stoi(argv[3]);
    const int vocab = std::stoi(argv[4]), rounds = std::stoi(argv[5]);
    if (batch <= 0 || sequence <= 0 || vocab <= 1 || rounds <= 0) return 2;
    const size_t count = static_cast<size_t>(batch) * sequence * vocab;
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) return 2;
    std::vector<uint16_t> values(count);
    uint32_t state = 1800;
    for (size_t i = 0; i < count; ++i) {
      state = state * 1664525u + 1013904223u;
      if (pattern == "random") {
        values[i] = state >> 16;
        if ((values[i] & 0x7fff) >= 0x7c00) values[i] &= ~0x0400;
      } else if (pattern == "ascending") {
        values[i] = (i % vocab) * 0x7bff / (vocab - 1);
      } else if (pattern == "periodic") {
        values[i] = (i % 101) * 0x100;
      } else return 2;
    }
    std::vector<float> converted;
    uint64_t checksum = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int repeat = 0; repeat < rounds; ++repeat) {
      values[0] = repeat % 2 == 0 ? 0x3c00 : 0xbc00;
      // Optional host allocation+copy models staging cost, NOT device readback.
      if (mode == "staged-copy") {
        const std::vector<uint16_t> staging(values);
        ConvertFp16ToFp32(staging, converted);
      } else {
        ConvertFp16ToFp32(values, converted);
      }
      auto ids = litert::lm::TopKTokenIds(converted, 1, batch, sequence);
      if (!ids.ok()) return 3;
      for (const auto& row : *ids) for (int id : row) checksum += id;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << elapsed << ' ' << checksum << '\n';
    return 0;
  }
  if (argc != 1) return 2;

  int batch, sequence;
  size_t count;
  while (std::cin >> batch >> sequence >> count) {
    // Guard the upstream function's integer arithmetic and conversion loop.
    // Invalid dimensions are not an upstream behavioral claim: passing them
    // directly to TopKTokenIds can divide by zero or overflow signed int.
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) return 2;
    std::vector<uint16_t> fp16_values(count);
    for (uint16_t& value : fp16_values) {
      uint32_t bits;
      if (!(std::cin >> std::hex >> bits >> std::dec) || bits > 0xffffu) return 2;
      value = static_cast<uint16_t>(bits);
    }
    if (batch <= 0 || sequence <= 0 ||
        batch > std::numeric_limits<int>::max() / sequence) {
      std::cout << "ERR\n";
      continue;
    }
    std::vector<float> logits;
    // Keep an allocated element for an empty input to avoid null pointer
    // arithmetic in upstream, while the supplied span remains empty.
    logits.reserve(count + 1);
    ConvertFp16ToFp32(fp16_values, logits);
    auto result = litert::lm::TopKTokenIds(logits, 1, batch, sequence);
    if (!result.ok()) {
      std::cout << "ERR\n";
      continue;
    }
    std::cout << "OK";
    for (const auto& row : *result) {
      for (int token : row) std::cout << ' ' << token;
    }
    std::cout << '\n';
  }
}
