// Copyright 2026 rt-rs contributors. SPDX-License-Identifier: Apache-2.0
// Diagnostic-only factory decoration. No production source or sampling changes.
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "nlohmann/json.hpp"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "runtime/components/prompt_template.h"
#include "runtime/components/sampler.h"
#include "runtime/components/sampler_factory.h"
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/io_types.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/sampler_params.pb.h"

namespace {
using nlohmann::json;
std::mutex report_mutex;
json report = {{"schema", 1}, {"kind", "instrumented_native_diagnostic_not_benchmark"},
               {"factory_calls", 0}, {"sample_calls", 0},
               {"factories", json::array()}, {"samples", json::array()},
               {"updates", json::array()}};

void Record(const std::string& key, const json& value) {
  std::lock_guard<std::mutex> lock(report_mutex);
  report[key] = value;
}

json SamplingConfig(const litert::lm::proto::SamplerParameters& params) {
  return {{"type", static_cast<int>(params.type())}, {"k", params.k()},
      {"p", params.p()}, {"temperature", params.temperature()},
      {"seed", params.seed()}, {"has_seed", params.has_seed()}};
}

class ObservedSampler final : public litert::lm::Sampler {
 public:
  explicit ObservedSampler(std::unique_ptr<litert::lm::Sampler> sampler, json sampling)
      : sampler_(std::move(sampler)), sampling_(std::move(sampling)) {}

  absl::Status SampleToIdAndScoreBuffer(
      const litert::TensorBuffer& logits, litert::TensorBuffer& ids,
      litert::TensorBuffer* scores) override {
    json sample;
    sample["sampling"] = sampling_;
    auto type = logits.TensorType();
    if (!type) return absl::InternalError("Probe could not inspect logits type");
    sample["dtype_enum"] = static_cast<int>(type->ElementType());
    sample["dtype"] = type->ElementType() == litert::ElementType::Float32 ? "float32" :
                      type->ElementType() == litert::ElementType::Float16 ? "float16" : "other";
    sample["shape"] = json::array();
    for (auto dim : type->Layout().Dimensions()) sample["shape"].push_back(dim);
    sample["scores_requested"] = scores != nullptr;
    // Sampling still executes exactly once in the unmodified native sampler.
    auto status = sampler_->SampleToIdAndScoreBuffer(logits, ids, scores);
    sample["status"] = status.ToString();
    if (status.ok()) {
      auto ids_type = ids.TensorType();
      auto size = ids.PackedSize();
      if (!ids_type || !size || ids_type->ElementType() != litert::ElementType::Int32 ||
          *size % sizeof(int32_t) != 0 || *size > 4096) {
        return absl::InternalError("Probe encountered unexpected output ID layout");
      }
      std::vector<int32_t> values(*size / sizeof(int32_t));
      if (!ids.Read(absl::MakeSpan(values)))
        return absl::InternalError("Probe could not read sampled IDs");
      sample["token_ids"] = values;
    }
    {
      std::lock_guard<std::mutex> lock(report_mutex);
      report["sample_calls"] = report["sample_calls"].get<int>() + 1;
      if (report["samples"].size() >= 64)
        return absl::ResourceExhaustedError("Probe exceeded bounded observation count");
      report["samples"].push_back(std::move(sample));
    }
    return status;
  }

  absl::Status UpdateConfig(const litert::lm::proto::SamplerParameters& params,
      int batch_size, std::shared_ptr<std::default_random_engine> random) override {
    auto status = sampler_->UpdateConfig(params, batch_size, std::move(random));
    if (status.ok()) sampling_ = SamplingConfig(params);
    std::lock_guard<std::mutex> lock(report_mutex);
    report["updates"].push_back({{"params", params.DebugString()},
                                {"sampling", SamplingConfig(params)},
                                {"proto_backend", static_cast<int>(params.backend())},
                                {"batch_size", batch_size}, {"status", status.ToString()}});
    return status;
  }
  bool CanHandleInput() const override { return sampler_->CanHandleInput(); }
  bool HandlesInput() const override { return sampler_->HandlesInput(); }
  absl::Status SetInferenceFuncAndInputTensors(
      int (*fn)(void*), void* arg, const litert::TensorBuffer* ids,
      const litert::TensorBuffer* prev_pos, const litert::TensorBuffer* pos,
      const litert::TensorBuffer* prev_mask, const litert::TensorBuffer* mask,
      const litert::TensorBuffer* prev_param, const litert::TensorBuffer* param) override {
    return sampler_->SetInferenceFuncAndInputTensors(
        fn, arg, ids, prev_pos, pos, prev_mask, mask, prev_param, param);
  }
 private:
  std::unique_ptr<litert::lm::Sampler> sampler_;
  json sampling_;
};
}  // namespace

// This Linux Itanium ABI name is pinned to sampler_factory.h. The type assertion
// protects the function signature; the workflow checks both real/wrapper symbols.
#define FACTORY_SYMBOL "_ZN6litert2lm13CreateSamplerENS0_7BackendEiNS0_5proto17SamplerParametersESt8optionalISt17reference_wrapperIKNS_11EnvironmentEEES4_IiESA_S4_INS0_18ActivationDataTypeEE"
using FactoryResult = absl::StatusOr<std::unique_ptr<litert::lm::Sampler>>;
using EnvironmentRef = std::optional<std::reference_wrapper<const litert::Environment>>;
extern "C" FactoryResult RealFactory(litert::lm::Backend, int,
    litert::lm::proto::SamplerParameters, EnvironmentRef, std::optional<int>,
    std::optional<int>, std::optional<litert::lm::ActivationDataType>)
    asm("__real_" FACTORY_SYMBOL);
extern "C" FactoryResult WrapFactory(litert::lm::Backend, int,
    litert::lm::proto::SamplerParameters, EnvironmentRef, std::optional<int>,
    std::optional<int>, std::optional<litert::lm::ActivationDataType>)
    asm("__wrap_" FACTORY_SYMBOL);
static_assert(std::is_same_v<decltype(&WrapFactory), decltype(&litert::lm::CreateSampler)>);

FactoryResult WrapFactory(litert::lm::Backend backend, int batch,
    litert::lm::proto::SamplerParameters params, EnvironmentRef env,
    std::optional<int> sequence, std::optional<int> vocab,
    std::optional<litert::lm::ActivationDataType> activation) {
  const auto params_text = params.DebugString();
  const json sampling = SamplingConfig(params);
  auto result = RealFactory(backend, batch, std::move(params), env, sequence, vocab, activation);
  {
    std::lock_guard<std::mutex> lock(report_mutex);
    report["factory_calls"] = report["factory_calls"].get<int>() + 1;
    report["factories"].push_back({{"backend", static_cast<int>(backend)},
        {"batch_size", batch}, {"params", params_text}, {"sampling", sampling}, {"status", result.status().ToString()}});
  }
  if (!result.ok()) return result.status();
  if (!*result) return absl::InternalError("Probe received a null sampler");
  return std::unique_ptr<litert::lm::Sampler>(new ObservedSampler(std::move(*result), sampling));
}

namespace {
absl::Status Run(const std::string& path, const std::string& cache_dir) {
  using namespace litert::lm;
  constexpr char prompt[] = "Complete this sentence in a few words: The sky is";
  Record("requested", {{"backend", "cpu"}, {"sampler_backend", "cpu"},
      {"threads", 2}, {"k", 1}, {"p", 1}, {"temperature", 1}, {"seed", 0},
      {"max_output_tokens", 16}, {"prompt", prompt}});
  ABSL_ASSIGN_OR_RETURN(auto assets, ModelAssets::Create(path));
  ABSL_ASSIGN_OR_RETURN(auto settings, EngineSettings::CreateDefault(
      std::move(assets), Backend::CPU, std::nullopt, std::nullopt, Backend::CPU));
  auto& executor = settings.GetMutableMainExecutorSettings();
  executor.SetCacheDir(cache_dir);
  ABSL_ASSIGN_OR_RETURN(auto cpu, executor.MutableBackendConfig<CpuConfig>());
  cpu.number_of_threads = 2;
  executor.SetBackendConfig(cpu);
  ABSL_ASSIGN_OR_RETURN(auto engine, EngineFactory::CreateDefault(std::move(settings)));
  std::ostringstream resolved;
  resolved << engine->GetEngineSettings();
  Record("resolved_engine_settings", resolved.str());
  ABSL_ASSIGN_OR_RETURN(auto resolved_cpu,
      engine->GetEngineSettings().GetMainExecutorSettings().GetBackendConfig<CpuConfig>());
  Record("resolved_threads", resolved_cpu.number_of_threads);
  auto session = SessionConfig::CreateDefault();
  session.SetSamplerBackend(Backend::CPU);
  session.SetMaxOutputTokens(16);
  session.SetEnableSpeculativeDecoding(false);
  auto& params = session.GetMutableSamplerParams();
  params.set_type(proto::SamplerParameters::TOP_P);
  params.set_k(1);
  params.set_p(1.0f);
  params.set_temperature(1.0f);
  params.set_seed(0);
  params.set_backend(proto::SamplerParameters::CPU);
  ABSL_ASSIGN_OR_RETURN(auto config, ConversationConfig::Builder()
      .SetSessionConfig(session).SetReturnErrorOnMaxTokensReached(false).Build(*engine));
  ABSL_ASSIGN_OR_RETURN(auto conversation, Conversation::Create(*engine, config));
  resolved.str("");
  resolved.clear();
  resolved << conversation->GetConfig().GetSessionConfig();
  Record("resolved_conversation_session_config", resolved.str());
  Record("resolved_max_output_tokens", conversation->GetConfig().GetSessionConfig().GetMaxOutputTokens());
  Record("resolved_prompt_template", std::string(conversation->GetConfig().GetPromptTemplate().GetTemplateSource()));
  // The Conversation API applies the actual model metadata/template. No manually
  // fabricated chat markers or tokenizer shortcuts are used.
  ABSL_ASSIGN_OR_RETURN(auto response, conversation->SendMessage(Message{
      {"role", "user"}, {"content", Message::array({{{"type", "text"}, {"text", prompt}}})}}));
  Record("response", json::parse(response.dump()));
  std::lock_guard<std::mutex> lock(report_mutex);
  if (report["factory_calls"].get<int>() <= 0 || report["sample_calls"].get<int>() <= 0)
    return absl::FailedPreconditionError("Factory/sample observer did not execute");
  return absl::OkStatus();
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: probe MODEL_PATH CACHE_DIR\n";
    return 2;
  }
  const auto status = Run(argv[1], argv[2]);
  Record("status", status.ToString());
  std::cout << report.dump(2) << '\n';
  return status.ok() ? 0 : 1;
}
