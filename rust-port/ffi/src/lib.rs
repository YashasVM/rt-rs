// Copyright 2026 rt-rs contributors.
// SPDX-License-Identifier: Apache-2.0
//! Audited C ABI boundary for the safe detector core. See native/stop_ffi.h.
//! Handles are exclusively owned, must not be used concurrently, and must not
//! alias any input/output storage. Pointer validation cannot detect dangling or
//! fabricated addresses: callers must uphold the documented allocation bounds.
//! Panics poison the handle and are caught before the ABI boundary. Allocator
//! aborts (including allocation failure) and process signals are not recoverable.
#![deny(unsafe_op_in_unsafe_fn)]

use rt_rs_stop::{Error, StopTokenDetector};
use std::num::NonZeroUsize;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

const OK: i32 = 0;
const INVALID: i32 = 3;
const INTERNAL: i32 = 13;

pub struct DetectorHandle {
    detector: StopTokenDetector,
    scratch: Vec<i32>,
    poisoned: bool,
}

fn error_code(error: Error) -> i32 {
    match error {
        Error::InvalidTokenRow => INTERNAL,
        _ => INVALID,
    }
}

// The caller guarantees live exclusive ownership of the handle, if non-null.
unsafe fn guard(handle: *mut DetectorHandle, op: impl FnOnce(&mut DetectorHandle) -> i32) -> i32 {
    // SAFETY: caller contract guarantees exclusive access to a live allocation.
    let Some(handle) = (unsafe { handle.as_mut() }) else {
        return INVALID;
    };
    if handle.poisoned {
        return INTERNAL;
    }
    match catch_unwind(AssertUnwindSafe(|| op(handle))) {
        Ok(status) => status,
        Err(_) => {
            handle.poisoned = true;
            INTERNAL
        }
    }
}

// Checks representable slice metadata, not allocation validity. The latter is
// part of the C caller contract. Empty buffers may use null pointers.
fn valid_buffer<T>(pointer: *const T, length: usize) -> bool {
    length == 0
        || (!pointer.is_null()
            && (pointer as usize) % std::mem::align_of::<T>() == 0
            && length <= isize::MAX as usize / std::mem::size_of::<T>()
            && (pointer as usize)
                .checked_add(length * std::mem::size_of::<T>())
                .is_some())
}

unsafe fn input<'a, T>(pointer: *const T, length: usize) -> Result<&'a [T], i32> {
    if !valid_buffer(pointer, length) {
        return Err(INVALID);
    }
    if length == 0 {
        return Ok(&[]);
    }
    // SAFETY: validated metadata; caller guarantees readable allocation/lifetime.
    Ok(unsafe { std::slice::from_raw_parts(pointer, length) })
}

/// # Safety
/// Returned pointer must be freed exactly once with rt_rs_stop_destroy. Batch
/// must be nonzero and fit the C++ signed index range. Null indicates failure.
#[no_mangle]
pub extern "C" fn rt_rs_stop_create(batch: usize) -> *mut DetectorHandle {
    if batch > i32::MAX as usize {
        return ptr::null_mut();
    }
    let Some(batch) = NonZeroUsize::new(batch) else {
        return ptr::null_mut();
    };
    catch_unwind(|| {
        Box::into_raw(Box::new(DetectorHandle {
            detector: StopTokenDetector::new(batch),
            scratch: vec![0; batch.get()],
            poisoned: false,
        }))
    })
    .unwrap_or(ptr::null_mut())
}

/// # Safety
/// Handle must be null or a live uniquely owned pointer returned by this API.
/// It must not be used again after this call, including by concurrent callers.
#[no_mangle]
pub unsafe extern "C" fn rt_rs_stop_destroy(handle: *mut DetectorHandle) {
    if !handle.is_null() {
        // SAFETY: caller transfers the unique allocation back exactly once.
        drop(unsafe { Box::from_raw(handle) });
    }
}

/// # Safety
/// Handle must remain live and not be mutated concurrently. The returned handle
/// is an independent deep copy and has the same destruction contract as create.
#[no_mangle]
pub unsafe extern "C" fn rt_rs_stop_clone(handle: *const DetectorHandle) -> *mut DetectorHandle {
    // SAFETY: caller guarantees a live shared reference for this call.
    let Some(handle) = (unsafe { handle.as_ref() }) else {
        return ptr::null_mut();
    };
    if handle.poisoned {
        return ptr::null_mut();
    }
    catch_unwind(AssertUnwindSafe(|| {
        Box::into_raw(Box::new(DetectorHandle {
            detector: handle.detector.clone(),
            scratch: handle.scratch.clone(),
            poisoned: false,
        }))
    }))
    .unwrap_or(ptr::null_mut())
}

/// # Safety
/// Handle follows the exclusive ownership contract. Tokens are readable for
/// length elements, properly aligned, and do not overlap the handle.
#[no_mangle]
pub unsafe extern "C" fn rt_rs_stop_add(
    handle: *mut DetectorHandle,
    tokens: *const i32,
    length: usize,
) -> i32 {
    // SAFETY: all pointer requirements are forwarded from the public contract.
    unsafe {
        guard(handle, |h| {
            if length > i32::MAX as usize {
                return INVALID;
            }
            let tokens = match input(tokens, length) {
                Ok(v) => v,
                Err(e) => return e,
            };
            h.detector
                .add_stop_sequence(tokens)
                .map_or_else(error_code, |_| OK)
        })
    }
}

/// # Safety
/// Handle follows the exclusive ownership contract. Zero retains its batch.
#[no_mangle]
pub unsafe extern "C" fn rt_rs_stop_reset(handle: *mut DetectorHandle, batch: usize) -> i32 {
    // SAFETY: caller guarantees exclusive live handle.
    unsafe {
        guard(handle, |h| {
            if batch > i32::MAX as usize {
                return INVALID;
            }
            let size = if batch == 0 { h.scratch.len() } else { batch };
            h.scratch.resize(size, 0);
            h.detector.reset_batch(NonZeroUsize::new(batch));
            OK
        })
    }
}

/// # Safety
/// Handle is live and exclusive. Return 0 means marked, 5 means not found.
#[no_mangle]
pub unsafe extern "C" fn rt_rs_stop_mark(handle: *mut DetectorHandle, token: i32) -> i32 {
    // SAFETY: caller guarantees exclusive live handle.
    unsafe {
        guard(handle, |h| {
            if h.detector.mark_for_rollback(token) {
                OK
            } else {
                5
            }
        })
    }
}

// SAFETY: forwards the public ABI buffer contract; h is live/exclusive.
unsafe fn process_impl(h: &mut DetectorHandle, tokens: *const i32, length: usize) -> i32 {
    // SAFETY: caller supplies readable/writable disjoint live buffers.
    unsafe {
        if length != h.scratch.len() {
            return INVALID;
        }
        let tokens = match input(tokens, length) {
            Ok(v) => v,
            Err(e) => return e,
        };
        h.detector
            .process_tokens(tokens)
            .map_or_else(error_code, |_| OK)
    }
}

/// # Safety
/// Handle is live and exclusive. Tokens are readable for length elements and
/// do not overlap the handle or its internal storage. Invalid sizes do not mutate.
#[no_mangle]
pub unsafe extern "C" fn rt_rs_stop_process(
    handle: *mut DetectorHandle,
    tokens: *const i32,
    length: usize,
) -> i32 {
    // SAFETY: caller supplies live handle and readable nonaliasing token storage.
    unsafe { guard(handle, |h| process_impl(h, tokens, length)) }
}

// SAFETY: forwards the public ABI buffer contract; h is live/exclusive.
unsafe fn rows_impl(
    h: &mut DetectorHandle,
    rows: *const *const i32,
    lengths: *const usize,
    count: usize,
) -> i32 {
    // SAFETY: caller supplies readable/writable disjoint live buffers.
    unsafe {
        if count != h.scratch.len() {
            return INVALID;
        }
        let lengths = match input(lengths, count) {
            Ok(v) => v,
            Err(e) => return e,
        };
        if lengths.iter().any(|&n| n != 1) {
            return INTERNAL;
        }
        let rows = match input(rows, count) {
            Ok(v) => v,
            Err(e) => return e,
        };
        if rows.iter().any(|&p| !valid_buffer(p, 1)) {
            return INVALID;
        }
        for (slot, &row) in h.scratch.iter_mut().zip(rows) {
            *slot = *row;
        }
        h.detector
            .process_tokens(&h.scratch)
            .map_or_else(error_code, |_| OK)
    }
}

/// # Safety
/// Handle is live/exclusive. Rows/lengths contain count readable elements; every
/// nonempty row points to its declared readable elements. Metadata and token
/// buffers must not overlap handle storage. No per-call allocation is performed.
#[no_mangle]
pub unsafe extern "C" fn rt_rs_stop_process_rows(
    handle: *mut DetectorHandle,
    rows: *const *const i32,
    lengths: *const usize,
    count: usize,
) -> i32 {
    // SAFETY: caller supplies the documented live metadata/token buffers.
    unsafe { guard(handle, |h| rows_impl(h, rows, lengths, count)) }
}

// SAFETY: forwards the public ABI buffer contract; h is live/exclusive.
unsafe fn snapshot_impl(
    h: &mut DetectorHandle,
    found: *mut u8,
    discard: *mut i32,
    partial: *mut i32,
    rollback: *mut u8,
    count: usize,
) -> i32 {
    // SAFETY: caller supplies readable/writable disjoint live buffers.
    unsafe {
        if count != h.scratch.len()
            || !valid_buffer(found, count)
            || !valid_buffer(discard, count)
            || !valid_buffer(partial, count)
            || !valid_buffer(rollback, count)
        {
            return INVALID;
        }
        for i in 0..count {
            if h.detector.steps_before_stop_tokens()[i] > i32::MAX as usize
                || h.detector.max_partial_stop_token_length(i).unwrap() > i32::MAX as usize
            {
                h.poisoned = true;
                return INTERNAL;
            }
        }
        for i in 0..count {
            found
                .add(i)
                .write(u8::from(h.detector.stop_tokens_found()[i]));
            discard
                .add(i)
                .write(h.detector.steps_before_stop_tokens()[i] as i32);
            partial
                .add(i)
                .write(h.detector.max_partial_stop_token_length(i).unwrap() as i32);
            rollback
                .add(i)
                .write(u8::from(h.detector.ends_with_rollback_token(i).unwrap()));
        }
        OK
    }
}

/// # Safety
/// Handle is live/exclusive. Each output is writable for count elements, aligned,
/// and disjoint from the handle and all other outputs. No pointers are retained.
/// Returns an error before writing when metadata is invalid; counter overflow
/// beyond the defined C++ range returns Internal and poisons the handle.
#[no_mangle]
pub unsafe extern "C" fn rt_rs_stop_snapshot(
    handle: *mut DetectorHandle,
    found: *mut u8,
    discard: *mut i32,
    partial: *mut i32,
    rollback: *mut u8,
    count: usize,
) -> i32 {
    // SAFETY: caller owns disjoint writable output arrays and the live handle.
    unsafe {
        guard(handle, |h| {
            snapshot_impl(h, found, discard, partial, rollback, count)
        })
    }
}

fn valid_outputs(
    found: *mut u8,
    discard: *mut i32,
    partial: *mut i32,
    rollback: *mut u8,
    count: usize,
) -> bool {
    valid_buffer(found, count)
        && valid_buffer(discard, count)
        && valid_buffer(partial, count)
        && valid_buffer(rollback, count)
}

/// # Safety
/// Same live/exclusive handle and input/output buffer contracts as process and
/// snapshot combined. Inputs may overlap outputs: all input borrows end when
/// process_impl returns, before snapshot_impl creates any writes.
/// Invalid metadata does not mutate detector state or write outputs.
#[no_mangle]
pub unsafe extern "C" fn rt_rs_stop_process_snapshot(
    handle: *mut DetectorHandle,
    tokens: *const i32,
    count: usize,
    found: *mut u8,
    discard: *mut i32,
    partial: *mut i32,
    rollback: *mut u8,
) -> i32 {
    // SAFETY: caller upholds both processing and snapshot pointer contracts.
    unsafe {
        guard(handle, |h| {
            if count != h.scratch.len() || !valid_outputs(found, discard, partial, rollback, count)
            {
                return INVALID;
            }
            let status = process_impl(h, tokens, count);
            if status != OK {
                return status;
            }
            snapshot_impl(h, found, discard, partial, rollback, count)
        })
    }
}

/// # Safety
/// Same contracts as process_rows and snapshot combined. Inputs may overlap
/// outputs: rows_impl finishes all metadata/token reads and drops their borrows
/// before snapshot_impl writes. Invalid metadata is atomic.
#[no_mangle]
pub unsafe extern "C" fn rt_rs_stop_rows_snapshot(
    handle: *mut DetectorHandle,
    rows: *const *const i32,
    lengths: *const usize,
    count: usize,
    found: *mut u8,
    discard: *mut i32,
    partial: *mut i32,
    rollback: *mut u8,
) -> i32 {
    // SAFETY: caller upholds both processing and snapshot pointer contracts.
    unsafe {
        guard(handle, |h| {
            if count != h.scratch.len() || !valid_outputs(found, discard, partial, rollback, count)
            {
                return INVALID;
            }
            let status = rows_impl(h, rows, lengths, count);
            if status != OK {
                return status;
            }
            snapshot_impl(h, found, discard, partial, rollback, count)
        })
    }
}

/// Select greedy k=1 indices into caller-owned contiguous output storage.
///
/// # Safety
/// Logits are readable for length f32 elements; output is writable for count
/// i32 elements. Both allocations are live/aligned and mutually disjoint.
/// Logits must not be concurrently mutated; output is exclusively writable
/// for the entire call (no concurrent reads or writes).
/// No pointer is retained. Null is allowed only for a zero-length input.
/// Shape/range/pointer metadata errors leave all outputs unchanged. Panics are
/// caught, but atomic output rollback is not promised for an unexpected panic.
#[no_mangle]
pub unsafe extern "C" fn rt_rs_greedy_token_ids(
    logits: *const f32,
    length: usize,
    batch: usize,
    sequence: usize,
    output: *mut i32,
    count: usize,
) -> i32 {
    let Some(positions) = batch.checked_mul(sequence) else { return INVALID; };
    if batch == 0
        || sequence == 0
        || positions > i32::MAX as usize
        || length > i32::MAX as usize
        || count != positions
        || length % positions != 0
        || !valid_buffer(logits, length)
        || !valid_buffer(output, count)
    {
        return INVALID;
    }
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: metadata checked above; caller guarantees live, readable
        // logits, disjoint writable outputs, and no concurrent mutation.
        unsafe {
            let logits = input(logits, length).expect("validated logits metadata");
            let vocab = length / positions;
            for position in 0..positions {
                let start = position * vocab;
                let id = rt_rs_stop::greedy::greedy_token_id(&logits[start..start + vocab]);
                // Bound follows length <= i32::MAX; empty vocab returns zero.
                output.add(position).write(id as i32);
            }
        }
        OK
    }))
    .unwrap_or(INTERNAL)
}

/// Select greedy k=1 indices directly from IEEE binary16 encodings.
/// No float32 staging allocation or pointer retention occurs.
///
/// # Safety
/// Logits are readable for length u16 binary16 encodings; output is writable for count
/// i32 elements. Both allocations are live/aligned and mutually disjoint.
/// Logits must not be concurrently mutated; output is exclusively writable
/// for the entire call (no concurrent reads or writes).
/// No pointer is retained. Null is allowed only for a zero-length input.
/// Shape/range/pointer metadata errors leave all outputs unchanged. Panics are
/// caught, but atomic output rollback is not promised for an unexpected panic.
#[no_mangle]
pub unsafe extern "C" fn rt_rs_greedy_fp16_token_ids(
    logits: *const u16,
    length: usize,
    batch: usize,
    sequence: usize,
    output: *mut i32,
    count: usize,
) -> i32 {
    let Some(positions) = batch.checked_mul(sequence) else { return INVALID; };
    if batch == 0
        || sequence == 0
        || positions > i32::MAX as usize
        || length > i32::MAX as usize
        || count != positions
        || length % positions != 0
        || !valid_buffer(logits, length)
        || !valid_buffer(output, count)
    {
        return INVALID;
    }
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: metadata checked above; caller guarantees live, readable
        // logits, disjoint writable outputs, and no concurrent mutation.
        unsafe {
            let logits = input(logits, length).expect("validated logits metadata");
            let vocab = length / positions;
            for position in 0..positions {
                let start = position * vocab;
                let id = rt_rs_stop::half::greedy_fp16_token_id(&logits[start..start + vocab]);
                // Bound follows length <= i32::MAX; empty vocab returns zero.
                output.add(position).write(id as i32);
            }
        }
        OK
    }))
    .unwrap_or(INTERNAL)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fused_metadata_errors_are_atomic_and_success_matches_split_calls() {
        // SAFETY: handles uniquely owned; all readable/writable arrays live and
        // disjoint. Null metadata is rejected before processing or writing.
        unsafe {
            let h = rt_rs_stop_create(2);
            assert_eq!(rt_rs_stop_add(h, [7].as_ptr(), 1), OK);
            let split = rt_rs_stop_clone(h);
            let mut found = [99u8; 2];
            let mut discard = [99i32; 2];
            let mut partial = [99i32; 2];
            let mut rollback = [99u8; 2];
            let tokens = [7, 0];
            assert_eq!(
                rt_rs_stop_process_snapshot(
                    h,
                    tokens.as_ptr(),
                    2,
                    found.as_mut_ptr(),
                    ptr::null_mut(),
                    partial.as_mut_ptr(),
                    rollback.as_mut_ptr()
                ),
                INVALID
            );
            assert_eq!((*h).detector.stop_tokens_found(), &[false, false]);
            assert_eq!(found, [99; 2]);
            let rows = [tokens.as_ptr(), tokens.as_ptr().add(1)];
            assert_eq!(
                rt_rs_stop_rows_snapshot(
                    h,
                    rows.as_ptr(),
                    [1, 0].as_ptr(),
                    2,
                    found.as_mut_ptr(),
                    discard.as_mut_ptr(),
                    partial.as_mut_ptr(),
                    rollback.as_mut_ptr()
                ),
                INTERNAL
            );
            assert_eq!((*h).detector.stop_tokens_found(), &[false, false]);
            assert_eq!(discard, [99; 2]);
            assert_eq!(
                rt_rs_stop_rows_snapshot(
                    h,
                    rows.as_ptr(),
                    [1, 1].as_ptr(),
                    2,
                    found.as_mut_ptr(),
                    discard.as_mut_ptr(),
                    partial.as_mut_ptr(),
                    rollback.as_mut_ptr()
                ),
                OK
            );
            assert_eq!(rt_rs_stop_process(split, tokens.as_ptr(), 2), OK);
            assert_eq!(
                (*h).detector.stop_tokens_found(),
                (*split).detector.stop_tokens_found()
            );
            assert_eq!(found, [1, 0]);
            assert_eq!(discard, [1, 0]);
            assert_eq!(
                rt_rs_stop_process_snapshot(
                    h,
                    tokens.as_ptr(),
                    2,
                    found.as_mut_ptr(),
                    discard.as_mut_ptr(),
                    partial.as_mut_ptr(),
                    rollback.as_mut_ptr()
                ),
                OK
            );
            assert_eq!(rt_rs_stop_process(split, tokens.as_ptr(), 2), OK);
            assert_eq!(
                (*h).detector.steps_before_stop_tokens(),
                (*split).detector.steps_before_stop_tokens()
            );
            assert_eq!(discard, [2, 0]);
            rt_rs_stop_destroy(h);
            rt_rs_stop_destroy(split);
        }
    }

    #[test]
    fn fused_inputs_may_overlap_outputs_after_input_borrows_end() {
        // SAFETY: input and discard intentionally share live storage; fused
        // processing ends its input borrows before any snapshot writes occur.
        unsafe {
            let h = rt_rs_stop_create(1);
            assert_eq!(rt_rs_stop_add(h, [7].as_ptr(), 1), OK);
            let mut discard = [7i32];
            let mut found = [0u8];
            let mut partial = [0i32];
            let mut rollback = [0u8];
            assert_eq!(
                rt_rs_stop_process_snapshot(
                    h,
                    discard.as_ptr(),
                    1,
                    found.as_mut_ptr(),
                    discard.as_mut_ptr(),
                    partial.as_mut_ptr(),
                    rollback.as_mut_ptr()
                ),
                OK
            );
            assert_eq!(discard, [1]);
            assert_eq!(found, [1]);
            assert_eq!(rt_rs_stop_reset(h, 0), OK);
            discard[0] = 7;
            let rows = [discard.as_ptr()];
            assert_eq!(
                rt_rs_stop_rows_snapshot(
                    h,
                    rows.as_ptr(),
                    [1].as_ptr(),
                    1,
                    found.as_mut_ptr(),
                    discard.as_mut_ptr(),
                    partial.as_mut_ptr(),
                    rollback.as_mut_ptr()
                ),
                OK
            );
            assert_eq!(discard, [1]);
            assert_eq!(found, [1]);
            rt_rs_stop_destroy(h);
        }
    }

    #[test]
    fn greedy_boundary_checks_metadata_and_preserves_reduction_semantics() {
        // SAFETY: valid arrays live/disjoint, invalid pointer metadata rejected
        // before any access; input contents remain immutable throughout calls.
        unsafe {
            let logits = [1.0, 3.0, 3.0, f32::NAN, 9.0, 1.0];
            let mut output = [99i32; 2];
            assert_eq!(
                rt_rs_greedy_token_ids(logits.as_ptr(), 6, 2, 1, output.as_mut_ptr(), 2),
                OK
            );
            assert_eq!(output, [1, 0]);
            for (len, batch, seq, count) in [
                (5, 2, 1, 2),
                (6, 0, 1, 2),
                (6, usize::MAX, 2, 2),
                (usize::MAX, 1, 1, 2),
                (6, 2, 1, 1),
            ] {
                output = [99; 2];
                assert_eq!(
                    rt_rs_greedy_token_ids(
                        logits.as_ptr(),
                        len,
                        batch,
                        seq,
                        output.as_mut_ptr(),
                        count
                    ),
                    INVALID
                );
                assert_eq!(output, [99; 2]);
            }
            assert_eq!(
                rt_rs_greedy_token_ids(ptr::null(), 6, 2, 1, output.as_mut_ptr(), 2),
                INVALID
            );
            assert_eq!(
                rt_rs_greedy_token_ids(logits.as_ptr(), 6, 2, 1, ptr::null_mut(), 2),
                INVALID
            );
            assert_eq!(output, [99; 2]);
            assert_eq!(
                rt_rs_greedy_token_ids(ptr::null(), 0, 2, 1, output.as_mut_ptr(), 2),
                OK
            );
            assert_eq!(output, [0; 2]);
        }
    }

    #[test]
    fn fp16_boundary_rejects_metadata_before_writes() {
        let logits = [0u16; 6];
        let mut output = [99i32; 2];
        // SAFETY: live disjoint buffers on success; every intentionally invalid
        // metadata case is rejected before forming a slice or writing output.
        unsafe {
            for (length, batch, sequence, count) in [
                (5, 2, 1, 2),
                (6, 0, 1, 2),
                (6, 2, 0, 2),
                (6, usize::MAX, 2, 2),
                (0, i32::MAX as usize + 1, 1, 2),
                (i32::MAX as usize + 1, 1, 1, 1),
                (6, 2, 1, 1),
                (6, 2, 1, 0),
            ] {
                assert_eq!(
                    rt_rs_greedy_fp16_token_ids(
                        logits.as_ptr(),
                        length,
                        batch,
                        sequence,
                        output.as_mut_ptr(),
                        count
                    ),
                    INVALID
                );
                assert_eq!(output, [99; 2]);
            }
            let misaligned_input = logits.as_ptr().cast::<u8>().add(1).cast::<u16>();
            let misaligned_output = output.as_mut_ptr().cast::<u8>().add(1).cast::<i32>();
            for input in [ptr::null(), misaligned_input] {
                assert_eq!(
                    rt_rs_greedy_fp16_token_ids(input, 6, 2, 1, output.as_mut_ptr(), 2),
                    INVALID
                );
                assert_eq!(output, [99; 2]);
            }
            for out in [ptr::null_mut(), misaligned_output] {
                assert_eq!(
                    rt_rs_greedy_fp16_token_ids(logits.as_ptr(), 6, 2, 1, out, 2),
                    INVALID
                );
                assert_eq!(output, [99; 2]);
            }
            assert_eq!(
                rt_rs_greedy_fp16_token_ids(ptr::null(), 0, 2, 1, output.as_mut_ptr(), 2),
                OK
            );
            assert_eq!(output, [0; 2]);
        }
    }

    fn check_fp16_boundary(logits: &[u16], batch: usize, sequence: usize) {
        let positions = batch * sequence;
        let vocab = logits.len() / positions;
        let converted: Vec<_> = logits
            .iter()
            .copied()
            .map(rt_rs_stop::half::half_to_f32)
            .collect();
        let mut output = vec![-1; positions];
        // SAFETY: immutable input and exclusive disjoint output remain live.
        assert_eq!(
            unsafe {
                rt_rs_greedy_fp16_token_ids(
                    logits.as_ptr(),
                    logits.len(),
                    batch,
                    sequence,
                    output.as_mut_ptr(),
                    positions,
                )
            },
            OK
        );
        for (position, actual) in output.into_iter().enumerate() {
            let start = position * vocab;
            assert_eq!(
                actual as usize,
                rt_rs_stop::greedy::greedy_token_id(&converted[start..start + vocab])
            );
        }
    }

    #[test]
    fn fp16_boundary_covers_every_encoding_and_tiled_nan_ties() {
        // Exercise every encoding as a seed and as a later competing value,
        // including signed zero, subnormals, infinities and both NaN classes.
        for bits in 0..=u16::MAX {
            check_fp16_boundary(&[bits, 0, 0x3c00, 0x8000, bits, 0x3c00], 2, 1);
        }
        for vocab in [1, 7, 8, 127, 128, 129, 255, 256, 257] {
            for index in 0..vocab {
                for special in [0x7c00, 0x7e00, 0x7c01, 0xfc00, 0x8000] {
                    let mut logits = vec![0x3c00; vocab * 4];
                    logits[index] = special;
                    logits[vocab + index] = 0x4000;
                    logits[vocab * 2 + index] = special;
                    logits[vocab * 2] = 0x7e01;
                    logits[vocab * 3 + index] = 0x4000;
                    logits[vocab * 4 - 1] = 0x4000;
                    check_fp16_boundary(&logits, 2, 2);
                }
            }
        }
    }

    #[test]
    fn invalid_metadata_and_clone_are_safe() {
        assert!(rt_rs_stop_create(0).is_null());
        // SAFETY: every non-null handle is owned and each buffer lives for call.
        unsafe {
            assert_eq!(rt_rs_stop_process(ptr::null_mut(), ptr::null(), 1), INVALID);
            let h = rt_rs_stop_create(1);
            assert_eq!(rt_rs_stop_add(h, [7].as_ptr(), 1), OK);
            assert_eq!(rt_rs_stop_process(h, ptr::null(), 1), INVALID);
            assert_eq!(rt_rs_stop_process(h, [7].as_ptr(), usize::MAX), INVALID);
            let copy = rt_rs_stop_clone(h);
            assert!(!copy.is_null());
            assert_eq!(rt_rs_stop_process(copy, [7].as_ptr(), 1), OK);
            assert!(!(*h).detector.all_done());
            assert!((*copy).detector.all_done());
            rt_rs_stop_destroy(copy);
            rt_rs_stop_destroy(h);
            rt_rs_stop_destroy(ptr::null_mut());
        }
    }
    #[test]
    fn invalid_later_row_does_not_mutate_detector() {
        // SAFETY: buffers are live, nonoverlapping; handles destroyed once.
        unsafe {
            let h = rt_rs_stop_create(2);
            rt_rs_stop_add(h, [7].as_ptr(), 1);
            let row = [7];
            let rows = [row.as_ptr(), ptr::null()];
            assert_eq!(
                rt_rs_stop_process_rows(h, rows.as_ptr(), [1, 0].as_ptr(), 2),
                INTERNAL
            );
            assert_eq!((*h).detector.stop_tokens_found(), &[false, false]);
            assert_eq!(
                rt_rs_stop_process_rows(h, rows.as_ptr(), [1, 1].as_ptr(), 2),
                INVALID
            );
            assert_eq!((*h).detector.stop_tokens_found(), &[false, false]);
            rt_rs_stop_destroy(h);
        }
    }
    #[test]
    fn snapshot_rejects_metadata_before_writing() {
        // SAFETY: all valid buffers are live/disjoint; invalid metadata is
        // rejected before any dereference or output writes.
        unsafe {
            let h = rt_rs_stop_create(1);
            rt_rs_stop_add(h, [7].as_ptr(), 1);
            rt_rs_stop_process(h, [7].as_ptr(), 1);
            let mut found = [99u8];
            let mut discard = [99i32];
            let mut partial = [99i32];
            let mut rollback = [99u8];
            assert_eq!(
                rt_rs_stop_snapshot(
                    h,
                    found.as_mut_ptr(),
                    discard.as_mut_ptr(),
                    ptr::null_mut(),
                    rollback.as_mut_ptr(),
                    1
                ),
                INVALID
            );
            assert_eq!(found, [99]);
            assert_eq!(discard, [99]);
            assert_eq!(rollback, [99]);
            assert_eq!(
                rt_rs_stop_snapshot(
                    h,
                    found.as_mut_ptr(),
                    discard.as_mut_ptr(),
                    partial.as_mut_ptr(),
                    rollback.as_mut_ptr(),
                    1
                ),
                OK
            );
            assert_eq!(found, [1]);
            assert_eq!(discard, [1]);
            assert_eq!(partial, [1]);
            assert_eq!(rollback, [0]);
            rt_rs_stop_destroy(h);
        }
    }

    #[test]
    fn panic_is_caught_and_handle_is_poisoned() {
        // SAFETY: handle is live/exclusive and destroyed once after poisoning.
        unsafe {
            let h = rt_rs_stop_create(1);
            assert_eq!(guard(h, |_| panic!("injected test panic")), INTERNAL);
            assert_eq!(rt_rs_stop_reset(h, 1), INTERNAL);
            assert!(rt_rs_stop_clone(h).is_null());
            rt_rs_stop_destroy(h);
        }
    }
}
