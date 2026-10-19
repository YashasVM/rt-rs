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
// Derived from LiteRT-LM v0.18.0 runtime/components/sampling_cpu_util.cc.

//! Standalone port of the `TopKTokenIds` greedy (`k = 1`) path.
//!
//! This does not implement general top-k, softmax, randomized sampling, or
//! native-runtime integration. Comparison order deliberately matches the C++
//! eight-lane reduction inside each 128-element tile, including its NaN effects.

use std::fmt;
use std::num::NonZeroUsize;

/// Invalid input shape or an output allocation that cannot be satisfied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GreedyError {
    /// The batch size multiplied by the sequence size overflows `usize`.
    ShapeOverflow,
    /// The logits length is not a multiple of batch size times sequence size.
    ShapeMismatch,
    /// The output's capacity is unrepresentable or its allocation failed.
    AllocationFailed,
}

impl fmt::Display for GreedyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::ShapeOverflow => "batch size times sequence size overflows usize",
            Self::ShapeMismatch => "logits length is not a multiple of batch times sequence size",
            Self::AllocationFailed => "unable to allocate greedy output indices",
        };
        f.write_str(message)
    }
}

impl std::error::Error for GreedyError {}

/// Select one vocabulary index for each batch/sequence position.
///
/// `logits` is laid out as `[batch][sequence][vocabulary]`; vocabulary size is
/// inferred from its length. The returned vector has `batch_size` rows, each
/// containing `sequence_size` indices. Empty logits represent an empty
/// vocabulary and produce zero at every position, matching the reference's
/// `MaxElement` behavior. Batch and sequence dimensions cannot be zero.
///
/// For inputs without NaNs, equal maxima retain the first index. NaNs are not filtered: the exact
/// reference reduction order is preserved, so this is not equivalent to a
/// scalar argmax for every input containing NaNs. In particular, a NaN at
/// vocabulary index zero fixes the result at zero; a NaN seeding a tile lane
/// masks later elements in that lane. Infinities use ordinary float comparisons.
///
/// Indices and dimensions use `usize`, rather than the reference's signed
/// integers. Parity is scoped to input dimensions and offsets within the C++
/// integer range; signed overflow and undefined native inputs are not emulated.
/// This implementation uses checked shape arithmetic and fallible output
/// reservations, and does not modify the logits.
pub fn greedy_token_ids(
    logits: &[f32],
    batch_size: NonZeroUsize,
    sequence_size: NonZeroUsize,
) -> Result<Vec<Vec<usize>>, GreedyError> {
    let batch_size = batch_size.get();
    let sequence_size = sequence_size.get();
    let positions = batch_size
        .checked_mul(sequence_size)
        .ok_or(GreedyError::ShapeOverflow)?;
    if logits.len() % positions != 0 {
        return Err(GreedyError::ShapeMismatch);
    }
    let vocab_size = logits.len() / positions;

    let mut output = Vec::new();
    output
        .try_reserve_exact(batch_size)
        .map_err(|_| GreedyError::AllocationFailed)?;
    let mut offset = 0;
    for _ in 0..batch_size {
        let mut row = Vec::new();
        row.try_reserve_exact(sequence_size)
            .map_err(|_| GreedyError::AllocationFailed)?;
        for _ in 0..sequence_size {
            // Validated shape guarantees every end is at most logits.len().
            let end = offset + vocab_size;
            row.push(greedy_token_id(&logits[offset..end]));
            offset = end;
        }
        output.push(row);
    }
    Ok(output)
}

const TILE_SIZE: usize = 128;
const LANES: usize = 8;

/// Literal comparison ordering of the reference's `Max<128>`.
fn tile_max(tile: &[f32]) -> f32 {
    debug_assert_eq!(tile.len(), TILE_SIZE);
    let mut maxima = [0.0; LANES];
    maxima.copy_from_slice(&tile[..LANES]);
    for group in tile[LANES..].chunks_exact(LANES) {
        for (maximum, &candidate) in maxima.iter_mut().zip(group) {
            // std::max(a, b) returns a unless a < b. f32::max has different
            // NaN semantics and must not be substituted here.
            if *maximum < candidate {
                *maximum = candidate;
            }
        }
    }
    let mut maximum = maxima[0];
    for &candidate in &maxima[1..] {
        // std::max_element retains the current value on unordered comparisons.
        if maximum < candidate {
            maximum = candidate;
        }
    }
    maximum
}

/// Select one index without allocating, preserving the reference `MaxElement`
/// comparison order, NaN masking and first-tie behavior. Empty input returns 0.
/// See [`greedy_token_ids`] for the complete semantic contract.
#[must_use]
pub fn greedy_token_id(logits: &[f32]) -> usize {
    let Some(&first) = logits.first() else {
        return 0;
    };
    let mut maximum = first;
    let mut maximum_index = 0;
    let mut offset = 0;
    let mut tiles = logits.chunks_exact(TILE_SIZE);
    for tile in &mut tiles {
        let candidate = tile_max(tile);
        if candidate > maximum {
            // The lane reduction does not retain indices. Search the tile in
            // original order, just as C++ does, to choose the first equal value.
            for (index, &value) in tile.iter().enumerate() {
                if value == candidate {
                    maximum = candidate;
                    maximum_index = offset + index;
                    break;
                }
            }
        }
        offset += TILE_SIZE;
    }
    for (index, &candidate) in tiles.remainder().iter().enumerate() {
        if candidate > maximum {
            maximum = candidate;
            maximum_index = offset + index;
        }
    }
    maximum_index
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nz(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).unwrap()
    }

    #[test]
    fn empty_vocabulary_produces_zero_indices() {
        assert_eq!(
            greedy_token_ids(&[], nz(2), nz(3)),
            Ok(vec![vec![0, 0, 0], vec![0, 0, 0]])
        );
    }

    #[test]
    fn rejects_mismatched_shape_without_changing_input() {
        let logits = [0.0, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            greedy_token_ids(&logits, nz(2), nz(2)),
            Err(GreedyError::ShapeMismatch)
        );
        assert_eq!(logits, [0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rejects_shape_overflow_before_allocation_or_division() {
        assert_eq!(
            greedy_token_ids(&[], nz(usize::MAX), nz(2)),
            Err(GreedyError::ShapeOverflow)
        );
        assert_eq!(
            greedy_token_ids(&[1.0], nz(2), nz(usize::MAX)),
            Err(GreedyError::ShapeOverflow)
        );
    }

    #[test]
    fn impossible_outer_or_inner_capacity_is_an_error() {
        assert_eq!(
            greedy_token_ids(&[], nz(usize::MAX), nz(1)),
            Err(GreedyError::AllocationFailed)
        );
        assert_eq!(
            greedy_token_ids(&[], nz(1), nz(usize::MAX)),
            Err(GreedyError::AllocationFailed)
        );
    }

    #[test]
    fn partitions_batch_sequence_and_vocabulary_dimensions() {
        let logits = [
            0.0, 3.0, 2.0, // batch 0, sequence 0
            5.0, 4.0, 3.0, // batch 0, sequence 1
            0.0, 1.0, 7.0, // batch 1, sequence 0
            1.0, 8.0, 8.0, // batch 1, sequence 1
        ];
        assert_eq!(
            greedy_token_ids(&logits, nz(2), nz(2)),
            Ok(vec![vec![1, 0], vec![2, 1]])
        );
    }

    #[test]
    fn tiled_reduction_starts_again_for_every_position() {
        let mut logits = vec![-1.0; 2 * 3 * 129];
        let expected = [0, 7, 127, 128, 63, 8];
        for (position, &peak) in expected.iter().enumerate() {
            logits[position * 129 + peak] = 10.0;
        }
        assert_eq!(
            greedy_token_ids(&logits, nz(2), nz(3)),
            Ok(vec![vec![0, 7, 127], vec![128, 63, 8]])
        );
    }

    #[test]
    fn every_peak_at_lane_and_tile_boundary_sizes_is_found() {
        for length in [1, 7, 8, 9, 127, 128, 129, 255, 256, 257, 383, 384, 385] {
            let mut logits = vec![-3.0; length];
            for peak in 0..length {
                logits[peak] = 2.0;
                assert_eq!(
                    greedy_token_id(&logits),
                    peak,
                    "length {length}, peak {peak}"
                );
                logits[peak] = -3.0;
            }
        }
    }

    #[test]
    fn ties_keep_first_index_within_and_across_tiles_and_tail() {
        for (first, second) in [(1, 8), (7, 8), (63, 127), (127, 128), (128, 256)] {
            let mut logits = vec![-1.0; 257];
            logits[first] = 9.0;
            logits[second] = 9.0;
            assert_eq!(greedy_token_id(&logits), first);
        }
        assert_eq!(greedy_token_id(&[5.0; 257]), 0);
    }

    #[test]
    fn signed_zero_ties_keep_first_index() {
        for first in [-0.0, 0.0] {
            let mut logits = vec![-first; 257];
            logits[0] = first;
            assert_eq!(greedy_token_id(&logits), 0);
        }
        let mut logits = vec![-1.0; 257];
        logits[7] = -0.0;
        logits[8] = 0.0;
        logits[128] = 0.0;
        assert_eq!(greedy_token_id(&logits), 7);
    }

    #[test]
    fn infinities_preserve_first_maximum() {
        assert_eq!(greedy_token_id(&[f32::NEG_INFINITY; 257]), 0);
        for peak in [0, 7, 127, 128, 256] {
            let mut logits = vec![f32::NEG_INFINITY; 257];
            logits[peak] = f32::INFINITY;
            logits[256] = f32::INFINITY;
            assert_eq!(greedy_token_id(&logits), peak);
        }
    }

    #[test]
    fn first_nan_pins_the_result_to_zero() {
        for length in [1, 127, 128, 129, 256, 257] {
            let mut logits = vec![f32::INFINITY; length];
            logits[0] = f32::NAN;
            assert_eq!(greedy_token_id(&logits), 0, "length {length}");
        }
    }

    #[test]
    fn scalar_nan_candidates_are_ignored() {
        assert_eq!(greedy_token_id(&[0.0, f32::NAN, 3.0, f32::NAN, 2.0]), 2);
        let mut logits = vec![0.0; 130];
        logits[128] = f32::NAN;
        logits[129] = 4.0;
        assert_eq!(greedy_token_id(&logits), 129);
    }

    #[test]
    fn nan_seeded_lane_masks_later_values_in_that_lane() {
        let mut logits = vec![0.0; 128];
        logits[2] = f32::NAN;
        logits[10] = f32::INFINITY;
        logits[3] = 5.0;
        assert_eq!(greedy_token_id(&logits), 3);
    }

    #[test]
    fn nan_in_first_lane_of_later_tile_masks_the_entire_tile() {
        let mut logits = vec![0.0; 256];
        logits[1] = 10.0;
        logits[128] = f32::NAN;
        logits[129] = f32::INFINITY;
        assert_eq!(greedy_token_id(&logits), 1);
    }

    #[test]
    fn nan_after_a_finite_lane_seed_does_not_poison_the_lane() {
        let mut logits = vec![0.0; 128];
        logits[8] = f32::NAN;
        logits[16] = 9.0;
        assert_eq!(greedy_token_id(&logits), 16);
    }

    #[test]
    fn equality_search_includes_values_hidden_by_a_nan_lane() {
        let mut logits = vec![0.0; 128];
        logits[1] = f32::NAN;
        logits[9] = 5.0; // Hidden in lane 1, but first equal value in the tile.
        logits[10] = 5.0; // Visible in lane 2 and selected by the reduction.
        assert_eq!(greedy_token_id(&logits), 9);
    }

    #[test]
    fn nan_masking_deliberately_differs_on_opposite_sides_of_tile_boundary() {
        let mut logits = vec![0.0; 128];
        logits[1] = f32::NAN;
        logits[9] = 10.0;
        logits[2] = 5.0;
        assert_eq!(greedy_token_id(&logits[..127]), 9); // Scalar remainder only.
        assert_eq!(greedy_token_id(&logits), 2); // Eight-lane reduction.
    }

    #[test]
    fn quiet_and_signaling_nan_bit_patterns_follow_comparison_order() {
        for bits in [0x7fc0_0001, 0xffc0_0001, 0x7f80_0001, 0xff80_0001] {
            let nan = f32::from_bits(bits);
            assert!(nan.is_nan());
            assert_eq!(greedy_token_id(&[nan, f32::INFINITY]), 0);
            let mut logits = vec![0.0; 128];
            logits[1] = nan;
            logits[9] = f32::INFINITY;
            logits[2] = 4.0;
            assert_eq!(greedy_token_id(&logits), 2);
        }
    }

    #[test]
    fn singleton_vocabulary_always_returns_zero() {
        let logits = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0];
        assert_eq!(
            greedy_token_ids(&logits, nz(2), nz(2)),
            Ok(vec![vec![0, 0], vec![0, 0]])
        );
    }
}
