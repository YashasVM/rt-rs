// Copyright 2025 The ODML Authors.
// Copyright 2026 rt-rs contributors.
// SPDX-License-Identifier: Apache-2.0
//! Safe binary16 conversion and fused greedy selection without float32 staging.
//! The reduction preserves the eight-lane/128-element reference comparison order,
//! including NaN-seeded lane masking and signed-zero ties. No quantization occurs.
use crate::greedy::GreedyError;
use std::num::NonZeroUsize;

/// Exact binary16 to binary32 conversion, preserving sign and payload while
/// quieting signaling NaNs, as the pinned native portable conversion does.
#[must_use]
pub fn half_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 31);
    let fraction = u32::from(bits & 0x03ff);
    let magnitude = match exponent {
        0 if fraction == 0 => 0,
        0 => {
            let shift = fraction.leading_zeros() - 21;
            ((113 - shift) << 23) | (((fraction << shift) & 0x03ff) << 13)
        }
        31 => 0x7f80_0000 | (fraction << 13) | if fraction != 0 { 0x0040_0000 } else { 0 },
        _ => ((exponent + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(sign | magnitude)
}

fn is_nan(bits: u16) -> bool {
    bits & 0x7fff > 0x7c00
}

// Finite numbers and infinities have monotone keys; normalize signed zero so
// equal maxima preserve the first zero regardless of its sign. NaN keys are
// never used in comparisons, except a zero sentinel for permanently masked lanes.
fn order_key(bits: u16) -> u16 {
    if bits & 0x7fff == 0 {
        0x8000
    } else if bits & 0x8000 != 0 {
        !bits
    } else {
        bits ^ 0x8000
    }
}

fn tile_max_key(tile: &[u16]) -> Option<u16> {
    // A NaN in lane zero seeds std::max_element's accumulator: the whole
    // tile is masked, even if later lanes contain larger finite values.
    if is_nan(tile[0]) {
        return None;
    }
    let mut keys = [0u16; 8];
    let mut masked = [false; 8];
    for i in 0..8 {
        masked[i] = is_nan(tile[i]);
        if !masked[i] {
            keys[i] = order_key(tile[i]);
        }
    }
    for group in tile[8..].chunks_exact(8) {
        for lane in 0..8 {
            let bits = group[lane];
            if !masked[lane] && !is_nan(bits) {
                keys[lane] = keys[lane].max(order_key(bits));
            }
        }
    }
    // Every valid key is >0, so sentinel keys cannot beat lane zero.
    keys.into_iter().max()
}

/// Greedy index directly from binary16 storage, with no allocation or float32
/// staging. Empty input returns zero. NaNs/ties follow [`crate::greedy`].
#[must_use]
pub fn greedy_fp16_token_id(logits: &[u16]) -> usize {
    let Some(&first) = logits.first() else { return 0; };
    if is_nan(first) {
        return 0;
    }
    let mut maximum = order_key(first);
    let mut maximum_index = 0;
    let mut offset = 0;
    let mut tiles = logits.chunks_exact(128);
    for tile in &mut tiles {
        if let Some(candidate) = tile_max_key(tile) {
            if candidate > maximum {
                // Match native equality search, including values in a masked
                // lane whose later elements were ignored by the lane reduction.
                for (index, &bits) in tile.iter().enumerate() {
                    if !is_nan(bits) && order_key(bits) == candidate {
                        maximum = candidate;
                        maximum_index = offset + index;
                        break;
                    }
                }
            }
        }
        offset += 128;
    }
    for (index, &bits) in tiles.remainder().iter().enumerate() {
        if !is_nan(bits) && order_key(bits) > maximum {
            maximum = order_key(bits);
            maximum_index = offset + index;
        }
    }
    maximum_index
}

/// Select nested indices for `[batch][sequence][vocabulary]` binary16 logits.
/// The input is never modified; output allocations are fallible. The algorithm
/// does not allocate a converted float32 vocabulary buffer.
pub fn greedy_fp16_token_ids(
    logits: &[u16],
    batch: NonZeroUsize,
    sequence: NonZeroUsize,
) -> Result<Vec<Vec<usize>>, GreedyError> {
    let positions = batch
        .get()
        .checked_mul(sequence.get())
        .ok_or(GreedyError::ShapeOverflow)?;
    if logits.len() % positions != 0 {
        return Err(GreedyError::ShapeMismatch);
    }
    let vocab = logits.len() / positions;
    let mut output = Vec::new();
    output
        .try_reserve_exact(batch.get())
        .map_err(|_| GreedyError::AllocationFailed)?;
    let mut offset = 0;
    for _ in 0..batch.get() {
        let mut row = Vec::new();
        row.try_reserve_exact(sequence.get())
            .map_err(|_| GreedyError::AllocationFailed)?;
        for _ in 0..sequence.get() {
            row.push(greedy_fp16_token_id(&logits[offset..offset + vocab]));
            offset += vocab;
        }
        output.push(row);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::greedy::greedy_token_id;

    #[test]
    fn conversion_matches_exact_binary16_arithmetic() {
        for bits in 0..=u16::MAX {
            let exp = (bits >> 10) & 31;
            let fraction = bits & 1023;
            let actual = half_to_f32(bits);
            if exp == 31 && fraction != 0 {
                assert!(actual.is_nan());
                assert_eq!(actual.to_bits() & 0x0040_0000, 0x0040_0000);
                continue;
            }
            let magnitude = if exp == 31 {
                f64::INFINITY
            } else if exp == 0 {
                f64::from(fraction) * 2f64.powi(-24)
            } else {
                f64::from(1024 + fraction) * 2f64.powi(i32::from(exp) - 25)
            };
            let expected = if bits & 0x8000 == 0 {
                magnitude
            } else {
                -magnitude
            } as f32;
            assert_eq!(actual.to_bits(), expected.to_bits(), "{bits:04x}");
        }
    }

    #[test]
    fn every_half_value_preserves_tiled_selection() {
        let mut values = vec![0xbc00; 129];
        for bits in 0..=u16::MAX {
            values.fill(0xbc00);
            values[usize::from(bits) % 129] = bits;
            values[100] = 0x3c00;
            if bits % 3 == 0 {
                values[7] = 0x7c01;
                values[15] = 0x7bff;
            }
            let converted: Vec<_> = values.iter().copied().map(half_to_f32).collect();
            assert_eq!(
                greedy_fp16_token_id(&values),
                greedy_token_id(&converted),
                "{bits:04x}"
            );
        }
    }

    #[test]
    fn zero_nan_and_tile_boundary_semantics() {
        assert_eq!(greedy_fp16_token_id(&[]), 0);
        assert_eq!(greedy_fp16_token_id(&[0x8000, 0]), 0);
        assert_eq!(greedy_fp16_token_id(&[0, 0x8000]), 0);
        assert_eq!(greedy_fp16_token_id(&[0x7c01, 0x7c00]), 0);
        for length in [7, 8, 9, 127, 128, 129, 255, 256, 257] {
            for position in 0..length {
                let mut values = vec![0xbc00; length];
                values[position] = 0x7c00;
                assert_eq!(greedy_fp16_token_id(&values), position);
                values[position] = 0xfc01;
                let converted: Vec<_> = values.iter().copied().map(half_to_f32).collect();
                assert_eq!(greedy_fp16_token_id(&values), greedy_token_id(&converted));
            }
        }
    }

    #[test]
    fn shape_checks_and_empty_vocabulary_match_float_path() {
        let two = NonZeroUsize::new(2).unwrap();
        let one = NonZeroUsize::new(1).unwrap();
        let maximum = NonZeroUsize::new(usize::MAX).unwrap();
        assert_eq!(
            greedy_fp16_token_ids(&[], maximum, one),
            Err(GreedyError::AllocationFailed)
        );
        assert_eq!(
            greedy_fp16_token_ids(&[], one, maximum),
            Err(GreedyError::AllocationFailed)
        );
        assert_eq!(
            greedy_fp16_token_ids(&[], two, two),
            Ok(vec![vec![0, 0], vec![0, 0]])
        );
        assert_eq!(
            greedy_fp16_token_ids(&[0], two, two),
            Err(GreedyError::ShapeMismatch)
        );
        assert_eq!(
            greedy_fp16_token_ids(&[], NonZeroUsize::new(usize::MAX).unwrap(), two),
            Err(GreedyError::ShapeOverflow)
        );
    }
}
