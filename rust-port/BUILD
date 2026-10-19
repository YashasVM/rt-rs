# Copyright 2026 rt-rs contributors.
# SPDX-License-Identifier: Apache-2.0

load("@rules_rust//rust:defs.bzl", "rust_library", "rust_static_library", "rust_test")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

# Compile the dependency-free safe core directly with Bazel, not via Cargo.
rust_library(
    name = "rt_rs_stop",
    srcs = [
        "src/greedy.rs",
        "src/half.rs",
        "src/lib.rs",
    ],
    crate_name = "rt_rs_stop",
    crate_root = "src/lib.rs",
    edition = "2021",
    rustc_flags = ["-Cpanic=unwind"],
)

# rust_static_library provides CcInfo, including the Rust runtime linkage.
# Keep unwinding enabled so the ABI's catch_unwind guards retain their contract.
rust_static_library(
    name = "rt_rs_stop_ffi",
    srcs = ["ffi/src/lib.rs"],
    crate_name = "rt_rs_stop_ffi",
    crate_root = "ffi/src/lib.rs",
    edition = "2021",
    rustc_flags = ["-Cpanic=unwind"],
    deps = [":rt_rs_stop"],
)

# Only the opted-in sampler depends on this wrapper; no global link flags or
# propagated preprocessor definitions change other native targets.
cc_library(
    name = "greedy_rust",
    srcs = [
        "native/greedy_rust.cc",
        "native/stop_ffi.h",
    ],
    hdrs = ["native/greedy_rust.h"],
    visibility = ["//runtime/components:__pkg__"],
    deps = [
        ":rt_rs_stop_ffi",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

rust_test(
    name = "rt_rs_stop_test",
    crate = ":rt_rs_stop",
)

# The invocation test checks its forwarding signature against the real ABI.
# Header-only: the native-OFF test must not acquire a Rust link dependency.
cc_library(
    name = "greedy_ffi_test_headers",
    testonly = True,
    hdrs = ["native/stop_ffi.h"],
    visibility = ["//runtime/components:__pkg__"],
)
