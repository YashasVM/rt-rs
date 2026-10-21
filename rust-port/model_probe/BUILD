# Copyright 2026 rt-rs contributors. SPDX-License-Identifier: Apache-2.0
package(default_visibility = ["//visibility:private"])
licenses(["notice"])

# Linux-only diagnostic binary. No wrapping or instrumentation is propagated to
# production targets. Static linking is essential for GNU --wrap interception.
cc_binary(
    name = "probe",
    testonly = True,
    srcs = ["probe.cc"],
    linkstatic = True,
    linkopts = [
        "-Wl,--export-dynamic-symbol=LiteRt*",
        "-Wl,--wrap=_ZN6litert2lm13CreateSamplerENS0_7BackendEiNS0_5proto17SamplerParametersESt8optionalISt17reference_wrapperIKNS_11EnvironmentEEES4_IiESA_S4_INS0_18ActivationDataTypeEE",
    ],
    deps = [
        "//runtime/components:prompt_template",
        "//runtime/components:sampler",
        "//runtime/components:sampler_factory",
        "//runtime/conversation",
        "//runtime/conversation:io_types",
        "//runtime/engine:engine_factory",
        "//runtime/engine:engine_impl_selected",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:llm_executor_settings",
        "//runtime/proto:sampler_params_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_tensor_buffer",
        "@nlohmann_json//:json",
    ],
)
