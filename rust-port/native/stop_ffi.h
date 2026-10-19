// Copyright 2026 rt-rs contributors.
// SPDX-License-Identifier: Apache-2.0
#ifndef RT_RS_STOP_FFI_H_
#define RT_RS_STOP_FFI_H_
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Exclusive opaque ownership: create/clone -> exactly one destroy. Handles and
// buffers must be live, correctly aligned and sized. All input tokens, rows and
// lengths must be disjoint from the handle and all of its owned internal storage.
// No concurrent calls on one handle. Output arrays must be mutually disjoint
// and disjoint from the handle.
// No pointers are retained except the owned handle. Invalid/dangling handles or
// fabricated buffer addresses violate the caller contract. Null destroy is safe.
// Panics are caught and poison the handle: only destroy is then valid. Allocation
// aborts/signals are not recoverable. There are no callbacks/cancellation or tasks
// outliving any call. Copying a handle pointer does NOT clone ownership.
// Status codes use canonical Abseil numbers (0 OK, 3 Invalid, 5 NotFound,
// 13 Internal). Input validation errors leave detector state unchanged.
typedef struct RtRsStopDetector RtRsStopDetector;
RtRsStopDetector* rt_rs_stop_create(size_t batch);
RtRsStopDetector* rt_rs_stop_clone(const RtRsStopDetector* handle);
void rt_rs_stop_destroy(RtRsStopDetector* handle);
int32_t rt_rs_stop_add(RtRsStopDetector*, const int32_t*, size_t);
int32_t rt_rs_stop_reset(RtRsStopDetector*, size_t);
int32_t rt_rs_stop_mark(RtRsStopDetector*, int32_t);
int32_t rt_rs_stop_process(RtRsStopDetector*, const int32_t*, size_t);
int32_t rt_rs_stop_process_rows(RtRsStopDetector*, const int32_t* const*, const size_t*, size_t);
int32_t rt_rs_stop_snapshot(RtRsStopDetector*, uint8_t*, int32_t*, int32_t*, uint8_t*, size_t);
#ifdef __cplusplus
}
#endif
#endif
