// Copyright 2025 The ODML Authors.
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

#include "runtime/components/top_p_cpu_sampler.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/convert_tensor_buffer.h"  // IWYU pragma: keep
#include "tflite/types/half.h"  // from @litert

// This counter is test-local and observes the ABI call, not branch eligibility.
// Native OFF has no wrapper, __real reference, or Rust dependency; its zero
// counts are paired with the hosted workflow's Rust-symbol absence check.
namespace {
std::atomic<int> rust_fp16_calls{0};
}

#if defined(LITERTLM_TEST_WRAP_RUST_FP16)
#include "rust-port/native/stop_ffi.h"

extern "C" decltype(rt_rs_greedy_fp16_token_ids)
    __real_rt_rs_greedy_fp16_token_ids;

extern "C" int32_t __wrap_rt_rs_greedy_fp16_token_ids(
    const uint16_t* logits, size_t length, size_t batch, size_t sequence,
    int32_t* output, size_t count) {
  rust_fp16_calls.fetch_add(1, std::memory_order_relaxed);
  // Never replace the computation: --wrap resolves this to the actual Rust ABI.
  return __real_rt_rs_greedy_fp16_token_ids(logits, length, batch, sequence,
                                          output, count);
}
static_assert(std::is_same_v<decltype(rt_rs_greedy_fp16_token_ids),
                             decltype(__wrap_rt_rs_greedy_fp16_token_ids)>);
#endif

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::FloatNear;

Expected<TensorBuffer> CopyFp16ToTensorBuffer(absl::Span<const float> data,
                                              absl::Span<const int> dims) {
  std::vector<uint16_t> fp16_data;
  fp16_data.reserve(data.size());
  for (float val : data) {
    tflite::half half_val(val);
    uint16_t fp16_val;
    std::memcpy(&fp16_val, &half_val, sizeof(uint16_t));
    fp16_data.push_back(fp16_val);
  }

  LITERT_ASSIGN_OR_RETURN(
      auto tensor_buffer,
      TensorBuffer::CreateManagedHostMemory(
          RankedTensorType(ElementType::Float16,
                           Layout(Dimensions(dims.begin(), dims.end()))),
          fp16_data.size() * sizeof(uint16_t)));
  tensor_buffer.Write(absl::MakeConstSpan(fp16_data));
  return tensor_buffer;
}

TEST(TopPSamplerTest, Create) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/1, /*sequence_size=*/1,
                                        /*seed=*/1);
  EXPECT_TRUE(sampler_or.ok());
}

TEST(TopPSamplerTest, CreateWithZeroTemp) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/0.0,
                                        /*batch_size=*/1, /*sequence_size=*/1,
                                        /*seed=*/1);
  EXPECT_TRUE(sampler_or.ok());
}

TEST(TopPSamplerTest, CreateWithNegativeTemp) {
  auto sampler_or =
      TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/-1.0,
                          /*batch_size=*/1, /*sequence_size=*/1, /*seed=*/1);
  EXPECT_FALSE(sampler_or.ok());
  EXPECT_THAT(sampler_or.status().message(),
              testing::HasSubstr("Temperature must be >= 0"));
}

TEST(TopPSamplerTest, SampleToIdAndScoreBuffer_IdsOnly_BatchSize2) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/2, /*sequence_size=*/1,
                                        /*seed=*/1);
  ASSERT_TRUE(sampler_or.ok());
  auto sampler = *std::move(sampler_or);

  const std::vector<float> logits = {0.0, 0.0, 10.0, 0.0, 11.0, 12.0, 1.0, 2.0};
  auto logits_tensor = CopyToTensorBuffer<float>(logits, {2, 4});

  std::vector<int> ids_vector(2);
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {2});
  auto status = sampler->SampleToIdAndScoreBuffer(*logits_tensor, *ids_tensor,
                                                  /*scores_tensor=*/nullptr);
  EXPECT_TRUE(status.ok());

  auto ids = CopyFromTensorBuffer<int>(*ids_tensor);
  ASSERT_TRUE(ids.HasValue());
  // The sampled id is 2 and 1.
  EXPECT_THAT(*ids, ElementsAre(2, 1));
}

TEST(TopPSamplerTest, SampleToIdAndScoreBuffer_BatchSize2) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/2, /*sequence_size=*/1,
                                        /*seed=*/1);
  ASSERT_TRUE(sampler_or.ok());
  auto sampler = *std::move(sampler_or);

  const std::vector<float> logits = {
      std::numeric_limits<float>::min(), std::numeric_limits<float>::min(),
      std::numeric_limits<float>::max(), std::numeric_limits<float>::min(),
      std::numeric_limits<float>::min(), std::numeric_limits<float>::max(),
      std::numeric_limits<float>::min(), std::numeric_limits<float>::min()};
  auto logits_tensor = CopyToTensorBuffer<float>(logits, {2, 4});
  ASSERT_TRUE(logits_tensor.HasValue());

  std::vector<int> ids_vector(2);
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {2});
  ASSERT_TRUE(ids_tensor.HasValue());
  std::vector<float> scores_vector(2);
  auto scores_tensor =
      CopyToTensorBuffer<float>(absl::MakeConstSpan(scores_vector), {2});
  ASSERT_TRUE(scores_tensor.HasValue());
  auto status = sampler->SampleToIdAndScoreBuffer(*logits_tensor, *ids_tensor,
                                                  &(*scores_tensor));
  EXPECT_TRUE(status.ok());
  auto ids = CopyFromTensorBuffer<int>(*ids_tensor);
  ASSERT_TRUE(ids.HasValue());
  // The sampled id is 2 and 1.
  EXPECT_THAT(*ids, ElementsAre(2, 1));

  auto scores = CopyFromTensorBuffer<float>(*scores_tensor);
  ASSERT_TRUE(scores.HasValue());
  // The scores are the log of the probability of the sampled token.
  EXPECT_THAT(*scores, ElementsAre(std::log(1.0f), std::log(1.0f)));
}

TEST(TopPSamplerTest, SampleToIdAndScoreBuffer_BatchSize2SequenceLength2) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/2, /*sequence_size=*/2,
                                        /*seed=*/1);
  ASSERT_TRUE(sampler_or.ok());
  auto sampler = *std::move(sampler_or);

  const std::vector<float> logits = {
      0.0,  0.0,  10.0, 0.0,  // b0, s0: top is 2
      11.0, 0.0,  0.0,  0.0,  // b0, s1: top is 0
      0.0,  12.0, 0.0,  0.0,  // b1, s0: top is 1
      0.0,  0.0,  0.0,  13.0  // b1, s1: top is 3
  };
  auto logits_tensor = CopyToTensorBuffer<float>(logits, {2, 2, 4});
  ASSERT_TRUE(logits_tensor.HasValue());

  std::vector<int> ids_vector(4);
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {2, 2});
  ASSERT_TRUE(ids_tensor.HasValue());
  std::vector<float> scores_vector(4);
  auto scores_tensor =
      CopyToTensorBuffer<float>(absl::MakeConstSpan(scores_vector), {2, 2});
  ASSERT_TRUE(scores_tensor.HasValue());
  auto status = sampler->SampleToIdAndScoreBuffer(*logits_tensor, *ids_tensor,
                                                  &(*scores_tensor));
  EXPECT_TRUE(status.ok());
  auto ids = CopyFromTensorBuffer<int>(*ids_tensor);
  ASSERT_TRUE(ids.HasValue());
  // The sampled ids are 2, 0, 1, 3.
  EXPECT_THAT(*ids, ElementsAre(2, 0, 1, 3));

  auto scores = CopyFromTensorBuffer<float>(*scores_tensor);
  ASSERT_TRUE(scores.HasValue());
  // The scores are the log of the probability of the sampled token.
  EXPECT_THAT(*scores, ElementsAre(std::log(1.0f), std::log(1.0f),
                                   std::log(1.0f), std::log(1.0f)));
}

TEST(TopPSamplerTest, SampleToIdAndScoreBufferFp16_IdsOnly_BatchSize2) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/2, /*sequence_size=*/1,
                                        /*seed=*/1);
  ASSERT_TRUE(sampler_or.ok());
  auto sampler = *std::move(sampler_or);

  const std::vector<float> logits = {0.0, 0.0, 10.0, 0.0, 11.0, 12.0, 1.0, 2.0};
  auto logits_tensor = CopyFp16ToTensorBuffer(logits, {2, 4});
  ASSERT_TRUE(logits_tensor.HasValue());
  std::vector<int> ids_vector(2);
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {2});
  ASSERT_TRUE(ids_tensor.HasValue());
  auto status = sampler->SampleToIdAndScoreBuffer(*logits_tensor, *ids_tensor,
                                                  /*scores_tensor=*/nullptr);
  EXPECT_TRUE(status.ok());

  auto ids = CopyFromTensorBuffer<int>(*ids_tensor);
  ASSERT_TRUE(ids.HasValue());
  // The sampled id is 2 and 1.
  EXPECT_THAT(*ids, ElementsAre(2, 1));
}

TEST(TopPSamplerTest, SampleToIdAndScoreBufferFp16_BatchSize2) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/2, /*sequence_size=*/1,
                                        /*seed=*/1);
  ASSERT_TRUE(sampler_or.ok());
  auto sampler = *std::move(sampler_or);

  const std::vector<float> logits = {
      std::numeric_limits<float>::min(), std::numeric_limits<float>::min(),
      std::numeric_limits<float>::max(), std::numeric_limits<float>::min(),
      std::numeric_limits<float>::min(), std::numeric_limits<float>::max(),
      std::numeric_limits<float>::min(), std::numeric_limits<float>::min()};
  auto logits_tensor = CopyFp16ToTensorBuffer(logits, {2, 4});
  ASSERT_TRUE(logits_tensor.HasValue());

  std::vector<int> ids_vector(2);
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {2});
  ASSERT_TRUE(ids_tensor.HasValue());
  std::vector<float> scores_vector(2);
  auto scores_tensor =
      CopyToTensorBuffer<float>(absl::MakeConstSpan(scores_vector), {2});
  ASSERT_TRUE(scores_tensor.HasValue());

  auto status = sampler->SampleToIdAndScoreBuffer(*logits_tensor, *ids_tensor,
                                                  &(*scores_tensor));
  EXPECT_TRUE(status.ok());

  auto ids = CopyFromTensorBuffer<int>(*ids_tensor);
  ASSERT_TRUE(ids.HasValue());
  // The sampled id is 2 and 1.
  EXPECT_THAT(*ids, ElementsAre(2, 1));

  auto scores = CopyFromTensorBuffer<float>(*scores_tensor);
  ASSERT_TRUE(scores.HasValue());
  // The scores are the log of the probability of the sampled token.
  EXPECT_THAT(*scores, ElementsAre(std::log(1.0f), std::log(1.0f)));
}

TEST(TopPSamplerTest, SampleToIdAndScoreBufferFp16_BatchSize2SequenceLength2) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/2, /*sequence_size=*/2,
                                        /*seed=*/1);
  ASSERT_TRUE(sampler_or.ok());
  auto sampler = *std::move(sampler_or);

  const std::vector<float> logits = {
      0.0,  0.0,  10.0, 0.0,  // b0, s0: top is 2
      11.0, 0.0,  0.0,  0.0,  // b0, s1: top is 0
      0.0,  12.0, 0.0,  0.0,  // b1, s0: top is 1
      0.0,  0.0,  0.0,  13.0  // b1, s1: top is 3
  };
  auto logits_tensor = CopyFp16ToTensorBuffer(logits, {2, 2, 4});
  ASSERT_TRUE(logits_tensor.HasValue());

  std::vector<int> ids_vector(4);
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {2, 2});
  ASSERT_TRUE(ids_tensor.HasValue());
  std::vector<float> scores_vector(4);
  auto scores_tensor =
      CopyToTensorBuffer<float>(absl::MakeConstSpan(scores_vector), {2, 2});
  ASSERT_TRUE(scores_tensor.HasValue());

  auto status = sampler->SampleToIdAndScoreBuffer(*logits_tensor, *ids_tensor,
                                                  &(*scores_tensor));
  EXPECT_TRUE(status.ok());

  auto ids = CopyFromTensorBuffer<int>(*ids_tensor);
  ASSERT_TRUE(ids.HasValue());
  // The sampled ids are 2, 0, 1, 3.
  EXPECT_THAT(*ids, ElementsAre(2, 0, 1, 3));

  auto scores = CopyFromTensorBuffer<float>(*scores_tensor);
  ASSERT_TRUE(scores.HasValue());
  // The scores are the log of the probability of the sampled token.
  EXPECT_THAT(*scores, ElementsAre(std::log(1.0f), std::log(1.0f),
                                   std::log(1.0f), std::log(1.0f)));
}

TEST(TopPSamplerTest, UpdateConfig) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/1, /*sequence_size=*/1,
                                        /*seed=*/2);
  ASSERT_TRUE(sampler_or.ok());
  auto sampler = *std::move(sampler_or);

  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(1.0);
  sampler_params.set_temperature(100.0);

  auto status = sampler->UpdateConfig(sampler_params,
                                      /*batch_size=*/1, nullptr);
  ASSERT_TRUE(status.ok());

  const std::vector<float> logits = {0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0};
  auto logits_tensor = CopyToTensorBuffer<float>(logits, {1, 8});

  std::vector<int> ids_vector = {0, 1, 2, 3, 4, 5, 6, 7};
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {1, 8});
  ASSERT_TRUE(ids_tensor.HasValue());

  status = sampler->SampleToIdAndScoreBuffer(*logits_tensor, ids_tensor.Value(),
                                             /*scores_tensor=*/nullptr);
  ASSERT_TRUE(status.ok());

  auto ids = CopyFromTensorBuffer<int>(ids_tensor.Value());
  ASSERT_TRUE(ids.HasValue());
  // With topk=1, it should pick the argmax.
  EXPECT_EQ(ids.Value()[0], 4);

  // Update the config again.
  sampler_params.set_k(8);

  status = sampler->UpdateConfig(sampler_params, /*batch_size=*/1, nullptr);
  ASSERT_TRUE(status.ok());

  status = sampler->SampleToIdAndScoreBuffer(*logits_tensor, ids_tensor.Value(),
                                             /*scores_tensor=*/nullptr);
  ASSERT_TRUE(status.ok());

  ids = CopyFromTensorBuffer<int>(ids_tensor.Value());

  // With topk=8 and high temperature, it might pick a random token. With
  // seed=2, it should consistently not pick 4.
  EXPECT_NE(ids.Value()[0], 4);
}

TEST(TopPSamplerTest,
     SampleToIdAndScoreBuffer_GreedyScoresTensorNullAndNonNull) {
  auto sampler_or =
      TopPSampler::Create(/*k=*/1, /*p=*/1.0, /*temperature=*/0.0,
                          /*batch_size=*/1, /*sequence_size=*/1, /*seed=*/1);
  ASSERT_TRUE(sampler_or.ok());
  if (!sampler_or.ok()) {
    return;
  }
  auto sampler = std::move(*sampler_or);
  EXPECT_FALSE(sampler->ComputeExactLogProbs());

  // Vocabulary of size 4: logits = {0.0, 1.0, 2.0, 3.0}
  // Argmax token is 3.
  const std::vector<float> logits = {0.0f, 1.0f, 2.0f, 3.0f};
  auto logits_tensor = CopyToTensorBuffer<float>(logits, {1, 4});
  ASSERT_TRUE(logits_tensor.HasValue());

  // 1. Fast-path greedy decode: scores_tensor == nullptr
  std::vector<int> ids_vector_fast(1);
  auto ids_tensor_fast =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector_fast), {1});
  ASSERT_TRUE(ids_tensor_fast.HasValue());

  auto status_fast = sampler->SampleToIdAndScoreBuffer(
      *logits_tensor, *ids_tensor_fast, /*scores_tensor=*/nullptr);
  EXPECT_TRUE(status_fast.ok());

  auto ids_fast = CopyFromTensorBuffer<int>(*ids_tensor_fast);
  ASSERT_TRUE(ids_fast.HasValue());
  EXPECT_THAT(*ids_fast, ElementsAre(3));

  // 2. Decode with scores requested: default compute_exact_log_probs == false.
  // Preserves fast default Top-K score behavior (std::log(1.0f) = 0.0f).
  std::vector<int> ids_vector_scores(1);
  auto ids_tensor_scores =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector_scores), {1});
  ASSERT_TRUE(ids_tensor_scores.HasValue());

  std::vector<float> scores_vector(1);
  auto scores_tensor =
      CopyToTensorBuffer<float>(absl::MakeConstSpan(scores_vector), {1});
  ASSERT_TRUE(scores_tensor.HasValue());

  auto status_scores = sampler->SampleToIdAndScoreBuffer(
      *logits_tensor, *ids_tensor_scores, &(*scores_tensor));
  EXPECT_TRUE(status_scores.ok());

  auto ids_scores = CopyFromTensorBuffer<int>(*ids_tensor_scores);
  ASSERT_TRUE(ids_scores.HasValue());
  // Token ID selection is identical whether scores are requested or not.
  EXPECT_EQ(ids_fast.Value()[0], ids_scores.Value()[0]);
  EXPECT_THAT(*ids_scores, ElementsAre(3));

  auto scores_default = CopyFromTensorBuffer<float>(*scores_tensor);
  ASSERT_TRUE(scores_default.HasValue());
  EXPECT_THAT(*scores_default, ElementsAre(std::log(1.0f)));

  // 3. Opt-in exact full-vocabulary log-probabilities.
  // log_prob = 3.0 - (3.0 + ln(exp(-3) + exp(-2) + exp(-1) + exp(0)))
  //          = 3.0 - (3.0 + ln(1.5529918)) = -0.4401868
  sampler->SetComputeExactLogProbs(true);
  EXPECT_TRUE(sampler->ComputeExactLogProbs());

  status_scores = sampler->SampleToIdAndScoreBuffer(
      *logits_tensor, *ids_tensor_scores, &(*scores_tensor));
  EXPECT_TRUE(status_scores.ok());

  auto scores_exact = CopyFromTensorBuffer<float>(*scores_tensor);
  ASSERT_TRUE(scores_exact.HasValue());
  EXPECT_THAT(*scores_exact, ElementsAre(FloatNear(-0.44018677f, 1e-5)));
}

TEST(TopPSamplerTest, SampleToIdAndScoreBuffer_ExactLogProbsViaCreate) {
  auto sampler_or =
      TopPSampler::Create(/*k=*/1, /*p=*/1.0, /*temperature=*/0.0,
                          /*batch_size=*/1, /*sequence_size=*/1, /*seed=*/1,
                          /*compute_exact_log_probs=*/true);
  ASSERT_TRUE(sampler_or.ok());
  if (!sampler_or.ok()) {
    return;
  }
  auto sampler = std::move(*sampler_or);
  EXPECT_TRUE(sampler->ComputeExactLogProbs());

  const std::vector<float> logits = {0.0f, 1.0f, 2.0f, 3.0f};
  auto logits_tensor = CopyToTensorBuffer<float>(logits, {1, 4});
  ASSERT_TRUE(logits_tensor.HasValue());

  std::vector<int> ids_vector(1);
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {1});
  ASSERT_TRUE(ids_tensor.HasValue());

  std::vector<float> scores_vector(1);
  auto scores_tensor =
      CopyToTensorBuffer<float>(absl::MakeConstSpan(scores_vector), {1});
  ASSERT_TRUE(scores_tensor.HasValue());

  auto status = sampler->SampleToIdAndScoreBuffer(*logits_tensor, *ids_tensor,
                                                  &(*scores_tensor));
  EXPECT_TRUE(status.ok());

  auto scores = CopyFromTensorBuffer<float>(*scores_tensor);
  ASSERT_TRUE(scores.HasValue());
  EXPECT_THAT(*scores, ElementsAre(FloatNear(-0.44018677f, 1e-5)));
}


// Keep these tests unconditional: the same real TensorBuffer contract must hold
// with the native build and with --define=rt_rs_fp16_greedy=true. The FP32
// sampler is the upstream TopKTopPSampling oracle, not a test reimplementation
// of its nontrivial NaN/tie reduction order.
Expected<TensorBuffer> CopyFp16BitsToTensorBuffer(
    absl::Span<const uint16_t> bits, absl::Span<const int> dims) {
  LITERT_ASSIGN_OR_RETURN(
      auto tensor,
      TensorBuffer::CreateManagedHostMemory(
          RankedTensorType(ElementType::Float16,
                           Layout(Dimensions(dims.begin(), dims.end()))),
          bits.size() * sizeof(uint16_t)));
  LITERT_RETURN_IF_ERROR(tensor.Write(bits));
  return tensor;
}

std::vector<float> NativeHalfValues(absl::Span<const uint16_t> bits) {
  std::vector<float> values;
  values.reserve(bits.size());
  for (uint16_t word : bits) {
    tflite::half value;
    std::memcpy(&value, &word, sizeof(word));
    values.push_back(static_cast<float>(value));
  }
  return values;
}

class Fp16SamplerParityTest : public testing::Test {
 protected:
  void Reset(int batch_size = 1, int sequence_size = 1) {
    batch_size_ = batch_size;
    auto half = TopPSampler::Create(1, 1.0f, 1.0f, batch_size, sequence_size, 73);
    auto native = TopPSampler::Create(1, 1.0f, 1.0f, batch_size, sequence_size, 73);
    ASSERT_TRUE(half.ok());
    ASSERT_TRUE(native.ok());
    half_ = std::move(*half);
    native_ = std::move(*native);
    half_rng_ = std::make_shared<std::default_random_engine>(73);
    native_rng_ = std::make_shared<std::default_random_engine>(73);
    ASSERT_NO_FATAL_FAILURE(Configure(1, 1.0f, 1.0f));
  }

  void Configure(int k, float p, float temperature, bool exact = false) {
    proto::SamplerParameters params;
    params.set_k(k);
    params.set_p(p);
    params.set_temperature(temperature);
    params.set_compute_exact_log_probs(exact);
    ASSERT_TRUE(half_->UpdateConfig(params, batch_size_, half_rng_).ok());
    ASSERT_TRUE(native_->UpdateConfig(params, batch_size_, native_rng_).ok());
  }

  // Compare both successful outputs and error-time writes. In particular, the
  // existing sampler writes IDs before validating a supplied scores tensor.
  void Compare(const std::vector<uint16_t>& bits, const std::vector<int>& dims,
               const std::vector<int>& ids_dims,
               const std::vector<int>& scores_dims, bool expect_ok = true,
               bool request_scores = true, int expected_rust_calls = -1) {
    auto half_logits = CopyFp16BitsToTensorBuffer(bits, dims);
    auto native_logits = CopyToTensorBuffer<float>(
        NativeHalfValues(bits), Dimensions(dims.begin(), dims.end()));
    ASSERT_TRUE(half_logits.HasValue());
    ASSERT_TRUE(native_logits.HasValue());
    int ids_size = 1;
    for (int dim : ids_dims) ids_size *= dim;
    int scores_size = 1;
    for (int dim : scores_dims) scores_size *= dim;
    const std::vector<int> initial_ids(ids_size, -123);
    const std::vector<float> initial_scores(scores_size, -123.0f);
    auto half_ids = CopyToTensorBuffer<int>(
        initial_ids, Dimensions(ids_dims.begin(), ids_dims.end()));
    auto native_ids = CopyToTensorBuffer<int>(
        initial_ids, Dimensions(ids_dims.begin(), ids_dims.end()));
    auto half_scores = CopyToTensorBuffer<float>(
        initial_scores, Dimensions(scores_dims.begin(), scores_dims.end()));
    auto native_scores = CopyToTensorBuffer<float>(
        initial_scores, Dimensions(scores_dims.begin(), scores_dims.end()));
    ASSERT_TRUE(half_ids.HasValue());
    ASSERT_TRUE(native_ids.HasValue());
    ASSERT_TRUE(half_scores.HasValue());
    ASSERT_TRUE(native_scores.HasValue());
    const int before_half = rust_fp16_calls.load(std::memory_order_relaxed);
    const auto half_status = half_->SampleToIdAndScoreBuffer(
        *half_logits, *half_ids, request_scores ? &*half_scores : nullptr);
    const int after_half = rust_fp16_calls.load(std::memory_order_relaxed);
    if (expected_rust_calls >= 0) {
#if defined(LITERTLM_TEST_WRAP_RUST_FP16)
      EXPECT_EQ(after_half - before_half, expected_rust_calls);
#else
      EXPECT_EQ(after_half - before_half, 0);
#endif
    }
    const auto native_status = native_->SampleToIdAndScoreBuffer(
        *native_logits, *native_ids, request_scores ? &*native_scores : nullptr);
    EXPECT_EQ(rust_fp16_calls.load(std::memory_order_relaxed), after_half)
        << "FP32 reference must never call the Rust FP16 ABI";
    EXPECT_EQ(native_status.ok(), expect_ok) << native_status;
    EXPECT_EQ(half_status.code(), native_status.code());
    EXPECT_EQ(half_status.message(), native_status.message());
    EXPECT_TRUE(*half_rng_ == *native_rng_);

    auto half_ids_values = CopyFromTensorBuffer<int>(*half_ids);
    auto native_ids_values = CopyFromTensorBuffer<int>(*native_ids);
    auto half_scores_values = CopyFromTensorBuffer<float>(*half_scores);
    auto native_scores_values = CopyFromTensorBuffer<float>(*native_scores);
    ASSERT_TRUE(half_ids_values.HasValue());
    ASSERT_TRUE(native_ids_values.HasValue());
    ASSERT_TRUE(half_scores_values.HasValue());
    ASSERT_TRUE(native_scores_values.HasValue());
    EXPECT_EQ(*half_ids_values, *native_ids_values);
    ASSERT_EQ(half_scores_values->size(), native_scores_values->size());
    for (int i = 0; i < half_scores_values->size(); ++i) {
      if (std::isnan((*native_scores_values)[i])) {
        EXPECT_TRUE(std::isnan((*half_scores_values)[i]));
      } else {
        EXPECT_FLOAT_EQ((*half_scores_values)[i], (*native_scores_values)[i]);
      }
    }
    ids_ = *half_ids_values;
    scores_ = *half_scores_values;
    // The input must remain bit-identical, including signaling NaN payloads.
    std::vector<uint16_t> after(bits.size());
    ASSERT_TRUE(half_logits->Read(absl::MakeSpan(after)).HasValue());
    EXPECT_EQ(after, bits);
    retained_ids_.push_back(std::move(*half_ids));
    retained_scores_.push_back(std::move(*half_scores));
  }

  int batch_size_ = 1;
  std::unique_ptr<TopPSampler> half_;
  std::unique_ptr<TopPSampler> native_;
  std::shared_ptr<std::default_random_engine> half_rng_;
  std::shared_ptr<std::default_random_engine> native_rng_;
  std::vector<int> ids_;
  std::vector<float> scores_;
  std::vector<TensorBuffer> retained_ids_;
  std::vector<TensorBuffer> retained_scores_;
};

TEST_F(Fp16SamplerParityTest, ActualRustAbiInvocationAndNativeFallbacks) {
#if !defined(LITERTLM_TEST_LINUX_FP16_CALL_PROOF)
  GTEST_SKIP() << "Invocation proof requires the Linux GNU-compatible --wrap linker";
#else
  ASSERT_NO_FATAL_FAILURE(Reset());
  const std::vector<uint16_t> bits = {0x0000, 0x3c00, 0x4000, 0x4200};
  // Compare snapshots the FP16 call separately from its FP32 native oracle.
  // Each eligible call must enter the real Rust ABI exactly once when enabled.
  for (float temperature : {0.0f, 1.0f}) {
    ASSERT_NO_FATAL_FAILURE(Configure(1, 1.0f, temperature));
    ASSERT_NO_FATAL_FAILURE(
        Compare(bits, {1, 4}, {1}, {1}, true, true, 1));
    ASSERT_NO_FATAL_FAILURE(
        Compare(bits, {1, 4}, {1}, {1}, true, false, 1));
  }
  ASSERT_NO_FATAL_FAILURE(Configure(1, 1.0f, 1.0f, true));
  ASSERT_NO_FATAL_FAILURE(
      Compare(bits, {1, 4}, {1}, {1}, true, true, 0));
  // Exact-score configuration is still eligible when no scores are requested.
  ASSERT_NO_FATAL_FAILURE(
      Compare(bits, {1, 4}, {1}, {1}, true, false, 1));
  for (float temperature : {0.0f, 1.0f}) {
    ASSERT_NO_FATAL_FAILURE(Configure(2, 0.8f, temperature));
    ASSERT_NO_FATAL_FAILURE(
        Compare(bits, {1, 4}, {1}, {1}, true, true, 0));
  }
  // UpdateConfig accepts these configs; sampling must keep native validation.
  ASSERT_NO_FATAL_FAILURE(Configure(0, 1.0f, 1.0f));
  ASSERT_NO_FATAL_FAILURE(
      Compare(bits, {1, 4}, {1}, {1}, false, true, 0));
  ASSERT_NO_FATAL_FAILURE(Configure(1, -0.1f, 1.0f));
  ASSERT_NO_FATAL_FAILURE(
      Compare(bits, {1, 4}, {1}, {1}, false, true, 0));
  ASSERT_NO_FATAL_FAILURE(Configure(1, 1.0f, -1.0f));
  ASSERT_NO_FATAL_FAILURE(
      Compare(bits, {1, 4}, {1}, {1}, false, true, 0));
  ASSERT_NO_FATAL_FAILURE(
      Configure(1, std::numeric_limits<float>::quiet_NaN(), 1.0f));
  ASSERT_NO_FATAL_FAILURE(
      Compare(bits, {1, 4}, {1}, {1}, true, true, 0));
  ASSERT_NO_FATAL_FAILURE(Configure(1, 1.0f, 1.0f));
  // Bad logits/IDs are rejected before sampling. Bad scores are checked after
  // sampling and therefore must not suppress an otherwise eligible ABI call.
  ASSERT_NO_FATAL_FAILURE(
      Compare(bits, {2, 2}, {1}, {1}, false, true, 0));
  ASSERT_NO_FATAL_FAILURE(
      Compare(bits, {1, 4}, {2}, {1}, false, true, 0));
  ASSERT_NO_FATAL_FAILURE(
      Compare(bits, {1, 4}, {1}, {2}, false, true, 1));
  ASSERT_NO_FATAL_FAILURE(Reset(1, 2));
  ASSERT_NO_FATAL_FAILURE(
      Compare({0x0000, 0x3c00, 0x4000}, {1, 3}, {1, 2}, {1, 2},
              false, true, 0));
#endif
}

TEST_F(Fp16SamplerParityTest, RawBitsNaNsSignedZerosTiesAndInfinities) {
  ASSERT_NO_FATAL_FAILURE(Reset());
  const std::vector<std::vector<uint16_t>> cases = {
      {0x8000, 0x0000, 0xbc00},  // Equal signed zeros retain the first ID.
      {0x0000, 0x8000, 0xbc00},
      {0x3c00, 0x4000, 0x4000},
      {0x7e01, 0x7c00, 0x4000},  // NaN first element pins native ID to zero.
      {0xfe55, 0x4000, 0x7c00},
      {0x7c01, 0x4000, 0x3c00},  // Signaling NaN is supplied as raw bits.
      {0xbc00, 0x7e01, 0x4000},
      {0xfc00, 0x7c00, 0x7c00},
      {0xfc00, 0xfc00, 0xfc00},
      {0x8001, 0x0000, 0x0001},  // Subnormal ordering.
      {0xfbff, 0x7bff, 0x7bff}};
  for (const auto& bits : cases) {
    SCOPED_TRACE(testing::PrintToString(bits));
    for (float temperature : {0.0f, 1.0f}) {
      ASSERT_NO_FATAL_FAILURE(Configure(1, 1.0f, temperature));
      const auto rng_before = *half_rng_;
      ASSERT_NO_FATAL_FAILURE(Compare(bits, {1, 3}, {1}, {1}));
      EXPECT_THAT(scores_, ElementsAre(0.0f));
      EXPECT_TRUE(*half_rng_ == rng_before);
      const auto scored_ids = ids_;
      ASSERT_NO_FATAL_FAILURE(Compare(bits, {1, 3}, {1}, {1}, true, false));
      EXPECT_EQ(ids_, scored_ids);
      EXPECT_TRUE(*half_rng_ == rng_before);
    }
  }
}

TEST_F(Fp16SamplerParityTest, TileBoundariesAndNaNMaskedLanes) {
  ASSERT_NO_FATAL_FAILURE(Reset(/*batch_size=*/2, /*sequence_size=*/2));
  for (int size : {7, 8, 9, 127, 128, 129, 255, 256, 257}) {
    SCOPED_TRACE(size);
    std::vector<uint16_t> bits(4 * size, 0xbc00);
    // Four independent rows exercise tails, lane masks, and equality search.
    bits[size - 1] = 0x4000;
    bits[size + 1] = 0x7e01;
    bits[2 * size - 1] = 0x4200;
    bits[2 * size] = 0x7c01;
    bits[3 * size - 1] = 0x7c00;
    bits[3 * size + 1] = 0x4000;
    bits[4 * size - 1] = 0x4000;
    if (size >= 128) {
      bits[size + 9] = 0x7bff;  // Masked by the NaN at lane one.
      bits[size + 10] = 0x4200;
    }
    if (size >= 256) {
      bits[128] = 0x7e01;  // NaN at lane zero masks the second full tile.
      bits[129] = 0x7c00;
    }
    ASSERT_NO_FATAL_FAILURE(Compare(bits, {2, 2, size}, {2, 2}, {2, 2}));
    EXPECT_THAT(scores_, ElementsAre(0.0f, 0.0f, 0.0f, 0.0f));
  }
  // Native equality search can select a value in a masked lane if an unmasked
  // lane supplies the same maximum. A scalar argmax would miss other cases.
  ASSERT_NO_FATAL_FAILURE(Reset());
  std::vector<uint16_t> bits(128, 0xbc00);
  bits[1] = 0x7e01;
  bits[9] = 0x4000;
  bits[10] = 0x4000;
  ASSERT_NO_FATAL_FAILURE(Compare(bits, {1, 128}, {1}, {1}));
  EXPECT_THAT(ids_, ElementsAre(9));
}

TEST_F(Fp16SamplerParityTest, ExactScoresMatchNativeFullVocabularyFallback) {
  ASSERT_NO_FATAL_FAILURE(Reset(/*batch_size=*/2, /*sequence_size=*/2));
  const std::vector<uint16_t> bits = {
      0x0000, 0x3c00, 0x4000, 0x4200, 0xbc00, 0x0000, 0x3800, 0x3c00,
      0x4000, 0x4000, 0x3c00, 0x0000, 0x7c00, 0x3c00, 0xfc00, 0x0000};
  for (float temperature : {0.0f, 0.5f, 1.0f, 2.0f}) {
    SCOPED_TRACE(temperature);
    ASSERT_NO_FATAL_FAILURE(Configure(1, 1.0f, temperature, true));
    const auto rng_before = *half_rng_;
    ASSERT_NO_FATAL_FAILURE(Compare(bits, {2, 2, 4}, {2, 2}, {2, 2}));
    EXPECT_LT(scores_[0], 0.0f);  // Must not accidentally emit approximate zero.
    EXPECT_TRUE(*half_rng_ == rng_before);
    if (temperature == 0.0f) EXPECT_NEAR(scores_[0], -0.44018677f, 1e-5f);
    // Exact mode alone must not prevent ID-only calls from working.
    ASSERT_NO_FATAL_FAILURE(
        Compare(bits, {2, 2, 4}, {2, 2}, {2, 2}, true, false));
  }
}

TEST_F(Fp16SamplerParityTest, NonGreedyFallbackPreservesSeededRandomState) {
  ASSERT_NO_FATAL_FAILURE(Reset(/*batch_size=*/2, /*sequence_size=*/2));
  const std::vector<uint16_t> bits = {
      0x0000, 0x3400, 0x3800, 0x3c00, 0x3c00, 0x3800, 0x3400, 0x0000,
      0x3400, 0x3800, 0x3c00, 0x0000, 0x3800, 0x0000, 0x3c00, 0x3400};
  const auto initial_rng = *half_rng_;
  for (int k : {2, 4}) {
    for (float temperature : {0.0f, 1.0f}) {
      ASSERT_NO_FATAL_FAILURE(Configure(k, 0.8f, temperature));
      for (int draw = 0; draw < 16; ++draw) {
        SCOPED_TRACE(draw);
        ASSERT_NO_FATAL_FAILURE(Compare(bits, {2, 2, 4}, {2, 2}, {2, 2}));
      }
    }
  }
  EXPECT_FALSE(*half_rng_ == initial_rng);
}

TEST_F(Fp16SamplerParityTest, InvalidAndNonfiniteConfigsMatchNativeStatus) {
  ASSERT_NO_FATAL_FAILURE(Reset());
  struct Config {
    int k;
    float p;
    float temperature;
    bool valid;
  };
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  for (const auto& config : std::vector<Config>{
           {0, 1.0f, 1.0f, false}, {1, -0.1f, 1.0f, false},
           {1, 1.1f, 1.0f, false}, {1, 1.0f, -1.0f, false},
           {1, inf, 1.0f, false}, {1, 1.0f, -inf, false},
           {1, nan, 1.0f, true}, {1, 1.0f, nan, true},
           {1, 1.0f, inf, true}, {1, 0.0f, 0.0f, true}}) {
    SCOPED_TRACE(testing::Message() << "k=" << config.k << " p=" << config.p
                                  << " temperature=" << config.temperature);
    ASSERT_NO_FATAL_FAILURE(
        Configure(config.k, config.p, config.temperature));
    const auto rng_before = *half_rng_;
    ASSERT_NO_FATAL_FAILURE(Compare({0x0000, 0x3c00, 0x4000, 0x4200},
                                   {1, 4}, {1}, {1}, config.valid));
    EXPECT_TRUE(*half_rng_ == rng_before);
    if (!config.valid) {
      // Invalid sampling config wins over invalid scores shape, because the
      // native implementation samples before validating the scores tensor.
      ASSERT_NO_FATAL_FAILURE(Compare({0x0000, 0x3c00, 0x4000, 0x4200},
                                     {1, 4}, {1}, {2}, false));
      EXPECT_THAT(ids_, ElementsAre(-123));
    }
  }
  ASSERT_NO_FATAL_FAILURE(Configure(1, 1.0f, 1.0f));
  ASSERT_NO_FATAL_FAILURE(
      Compare({0x0000, 0x3c00, 0x4000, 0x4200}, {1, 4}, {1}, {1}));
  EXPECT_THAT(ids_, ElementsAre(3));
}

TEST_F(Fp16SamplerParityTest, InvalidShapesAndErrorTimeWritesMatchNative) {
  ASSERT_NO_FATAL_FAILURE(Reset(/*batch_size=*/1, /*sequence_size=*/2));
  ASSERT_NO_FATAL_FAILURE(
      Compare({0x0000, 0x3c00, 0x4000}, {1, 3}, {1, 2}, {1, 2}, false));
  const std::vector<uint16_t> bits(16, 0x3c00);
  ASSERT_NO_FATAL_FAILURE(Compare(bits, {2, 8}, {1, 2}, {1, 2}, false));
  ASSERT_NO_FATAL_FAILURE(Compare(bits, {1, 2, 2, 2, 2}, {1, 2}, {1, 2}, false));
  ASSERT_NO_FATAL_FAILURE(Compare(bits, {1, 2, 8}, {2}, {1, 2}, false));
  ASSERT_NO_FATAL_FAILURE(Compare(bits, {1, 2, 8}, {1, 2, 2, 2}, {1, 2}, false));
  ASSERT_NO_FATAL_FAILURE(Compare(bits, {1, 2, 8}, {1, 2}, {2}, false));
  EXPECT_THAT(ids_, ElementsAre(0, 0));  // IDs precede score validation.
  ASSERT_NO_FATAL_FAILURE(Compare(bits, {1, 2, 8}, {1, 2}, {1, 2, 2, 2}, false));
  EXPECT_THAT(ids_, ElementsAre(0, 0));
  ASSERT_NO_FATAL_FAILURE(Compare(bits, {1, 2, 8}, {1, 2}, {1, 2}));
}

TEST_F(Fp16SamplerParityTest, ConfigTransitionsAndOutputOwnership) {
  ASSERT_NO_FATAL_FAILURE(Reset());
  const std::vector<uint16_t> bits = {0x0000, 0x3c00, 0x4000, 0x4200};
  ASSERT_NO_FATAL_FAILURE(Compare(bits, {1, 4}, {1}, {1}));
  for (int iteration = 0; iteration < 8; ++iteration) {
    SCOPED_TRACE(iteration);
    ASSERT_NO_FATAL_FAILURE(Configure(4, 0.9f, 1.0f));
    ASSERT_NO_FATAL_FAILURE(Compare(bits, {1, 4}, {1}, {1}));
    ASSERT_NO_FATAL_FAILURE(Configure(1, 1.0f, 0.0f, true));
    ASSERT_NO_FATAL_FAILURE(Compare(bits, {1, 4}, {1}, {1}));
    EXPECT_LT(scores_[0], 0.0f);
    ASSERT_NO_FATAL_FAILURE(Configure(1, 0.5f, 1.0f));
    const auto rng_before = *half_rng_;
    ASSERT_NO_FATAL_FAILURE(Compare(bits, {1, 4}, {1}, {1}));
    EXPECT_THAT(ids_, ElementsAre(3));
    EXPECT_THAT(scores_, ElementsAre(0.0f));
    EXPECT_TRUE(*half_rng_ == rng_before);
  }
  // All input tensors have already been destroyed. Later calls and sampler
  // destruction must not invalidate or overwrite caller-owned earlier outputs.
  half_.reset();
  native_.reset();
  auto first_ids = CopyFromTensorBuffer<int>(retained_ids_.front());
  auto first_scores = CopyFromTensorBuffer<float>(retained_scores_.front());
  ASSERT_TRUE(first_ids.HasValue());
  ASSERT_TRUE(first_scores.HasValue());
  EXPECT_THAT(*first_ids, ElementsAre(3));
  EXPECT_THAT(*first_scores, ElementsAre(0.0f));
}

}  // namespace
}  // namespace litert::lm

