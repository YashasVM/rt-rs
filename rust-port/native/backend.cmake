# Copyright 2026 rt-rs contributors.
# SPDX-License-Identifier: Apache-2.0
# Standalone experiment only: never attach this to the production target graph.
function(rt_rs_link_standalone_adapter target root)
  if(CMAKE_CROSSCOMPILING OR NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
     NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    message(FATAL_ERROR "Rust stop backend currently supports native Linux x86_64 only; no Android/cross build is validated")
  endif()
  find_program(RT_RS_CARGO cargo REQUIRED)
  find_program(RT_RS_RUSTC rustc REQUIRED)
  execute_process(COMMAND "${RT_RS_RUSTC}" -vV OUTPUT_VARIABLE rust_version COMMAND_ERROR_IS_FATAL ANY)
  if(NOT rust_version MATCHES "host: x86_64-unknown-linux-gnu")
    message(FATAL_ERROR "Rust and C++ host targets must both be Linux x86_64 GNU")
  endif()
  set(cargo_target "${CMAKE_CURRENT_BINARY_DIR}/rt-rs-cargo")
  set(archive "${cargo_target}/release/librt_rs_stop_ffi.a")
  add_custom_command(OUTPUT "${archive}"
    COMMAND "${RT_RS_CARGO}" build --manifest-path "${root}/rust-port/ffi/Cargo.toml"
      --release --locked --offline --target-dir "${cargo_target}"
    DEPENDS "${root}/rust-port/ffi/Cargo.toml" "${root}/rust-port/ffi/Cargo.lock"
      "${root}/rust-port/ffi/src/lib.rs" "${root}/rust-port/Cargo.toml"
      "${root}/rust-port/src/lib.rs" "${root}/rust-port/src/greedy.rs"
    WORKING_DIRECTORY "${root}" VERBATIM)
  add_custom_target(rt_rs_stop_archive DEPENDS "${archive}")
  add_library(rt_rs_stop_ffi STATIC IMPORTED GLOBAL)
  set_target_properties(rt_rs_stop_ffi PROPERTIES IMPORTED_LOCATION "${archive}")
  add_dependencies(rt_rs_stop_ffi rt_rs_stop_archive)
  find_package(Threads REQUIRED)
  target_link_libraries(${target} PUBLIC rt_rs_stop_ffi Threads::Threads ${CMAKE_DL_LIBS} m)
  # This changes the class layout. PRIVATE would violate the ODR in consumers.
  target_compile_definitions(${target} PUBLIC LITERTLM_USE_RUST_STOP_DETECTOR=1)
  target_include_directories(${target} PUBLIC "${root}")
endfunction()
