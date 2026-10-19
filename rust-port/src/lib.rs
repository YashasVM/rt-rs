// Copyright 2025 The ODML Authors.
// Copyright 2026 rt-rs contributors.
// SPDX-License-Identifier: Apache-2.0
// Derived from LiteRT-LM v0.18.0 runtime/components/stop_token_detector.{h,cc}.

#![forbid(unsafe_code)]

pub mod greedy;

use std::num::NonZeroUsize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    EmptyStopSequence,
    BatchSizeMismatch,
    NoStopSequences,
    InvalidTokenRow,
}

/// Incremental token-ID stop detection with LiteRT-LM v0.18.0 semantics
/// plus singleton rollback marking from pinned current main (see README).
///
/// Each call consumes exactly one token per batch item. This owns all state
/// and does not call native code. Counts use usize instead of C++ signed int.
#[derive(Debug)]
pub struct StopTokenDetector {
    sequences: Vec<Vec<i32>>,
    rollback_marks: Vec<bool>,
    ends_with_rollback: Vec<bool>,
    progress: Vec<Vec<usize>>,
    partial: Vec<usize>,
    found: Vec<bool>,
    discard: Vec<usize>,
}

impl StopTokenDetector {
    pub fn new(batch_size: NonZeroUsize) -> Self {
        let mut detector = Self {
            sequences: Vec::new(),
            rollback_marks: Vec::new(),
            ends_with_rollback: Vec::new(),
            progress: Vec::new(),
            partial: Vec::new(),
            found: Vec::new(),
            discard: Vec::new(),
        };
        detector.reset_batch(Some(batch_size));
        detector
    }

    /// Duplicate sequences are successful no-ops, as in the C++ implementation.
    pub fn add_stop_sequence(&mut self, sequence: &[i32]) -> Result<(), Error> {
        if sequence.is_empty() {
            return Err(Error::EmptyStopSequence);
        }
        if !self.sequences.iter().any(|existing| existing == sequence) {
            self.sequences.push(sequence.to_vec());
            self.rollback_marks.push(false);
            for row in &mut self.progress {
                row.push(0);
            }
        }
        Ok(())
    }

    /// Current-main semantics: mark an existing singleton stop sequence for
    /// executor rollback. This only records state; no executor is called.
    /// Missing singleton sequences return false without changing state.
    pub fn mark_for_rollback(&mut self, token: i32) -> bool {
        if let Some(index) = self
            .sequences
            .iter()
            .position(|sequence| sequence == &[token])
        {
            self.rollback_marks[index] = true;
            true
        } else {
            false
        }
    }

    /// True only on the stopping step of a marked singleton sequence.
    /// Reset or processing another valid token clears the per-item flag.
    pub fn ends_with_rollback_token(&self, index: usize) -> Option<bool> {
        self.ends_with_rollback.get(index).copied()
    }

    /// Retains stop sequences; None retains the existing nonzero batch size.
    pub fn reset_batch(&mut self, batch_size: Option<NonZeroUsize>) {
        let size = batch_size.map_or(self.found.len(), NonZeroUsize::get);
        self.ends_with_rollback.resize(size, false);
        self.ends_with_rollback.fill(false);
        self.found.resize(size, false);
        self.found.fill(false);
        self.partial.resize(size, 0);
        self.partial.fill(0);
        self.discard.resize(size, 0);
        self.discard.fill(0);
        self.progress.resize_with(size, Vec::new);
        for row in &mut self.progress {
            row.resize(self.sequences.len(), 0);
            row.fill(0);
        }
    }

    pub fn process_tokens(&mut self, tokens: &[i32]) -> Result<(), Error> {
        if tokens.len() != self.found.len() {
            return Err(Error::BatchSizeMismatch);
        }
        self.process_validated_tokens(tokens.iter().copied())
    }

    /// Process one token per row, matching the native nested-vector overload.
    /// All row shapes are validated before state changes. Unlike native code,
    /// this does not allocate or copy an intermediate flattened token buffer.
    pub fn process_token_rows(&mut self, tokens: &[Vec<i32>]) -> Result<(), Error> {
        if tokens.len() != self.found.len() {
            return Err(Error::BatchSizeMismatch);
        }
        if tokens.iter().any(|row| row.len() != 1) {
            return Err(Error::InvalidTokenRow);
        }
        self.process_validated_tokens(tokens.iter().map(|row| row[0]))
    }

    fn process_validated_tokens(&mut self, tokens: impl Iterator<Item = i32>) -> Result<(), Error> {
        if self.sequences.is_empty() {
            return Err(Error::NoStopSequences);
        }
        let batch = tokens
            .zip(&mut self.found)
            .zip(&mut self.discard)
            .zip(&mut self.partial)
            .zip(&mut self.progress)
            .zip(&mut self.ends_with_rollback);
        for (((((token, found), discard), partial), row), rollback) in batch {
            if *found {
                *discard += 1;
                *rollback = false;
                continue;
            }
            let mut maximum = 0;
            for ((progress, sequence), &marked) in row
                .iter_mut()
                .zip(&self.sequences)
                .zip(&self.rollback_marks)
            {
                if sequence.get(*progress) == Some(&token) {
                    *progress += 1;
                } else {
                    *progress = usize::from(sequence.first() == Some(&token));
                }
                if *progress > 0 && *progress == sequence.len() {
                    *found = true;
                    *discard = sequence.len();
                    *rollback = marked;
                }
                maximum = maximum.max(*progress);
            }
            *partial = maximum;
        }
        Ok(())
    }

    pub fn all_done(&self) -> bool {
        self.found.iter().all(|&found| found)
    }

    pub fn stop_tokens_found(&self) -> &[bool] {
        &self.found
    }

    pub fn steps_before_stop_tokens(&self) -> &[usize] {
        &self.discard
    }

    /// Includes completed matches, matching the C++ code rather than its comment.
    pub fn max_partial_stop_token_length(&self, index: usize) -> Option<usize> {
        self.partial.get(index).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detector(size: usize, sequences: &[&[i32]]) -> StopTokenDetector {
        let mut d = StopTokenDetector::new(NonZeroUsize::new(size).unwrap());
        for s in sequences {
            d.add_stop_sequence(s).unwrap();
        }
        d
    }

    #[test]
    fn rollback_marks_survive_reset_but_step_flags_do_not() {
        let mut d = detector(2, &[&[7], &[1, 7]]);
        assert!(!d.mark_for_rollback(1));
        assert!(d.mark_for_rollback(7));
        d.process_tokens(&[7, 0]).unwrap();
        assert_eq!(d.ends_with_rollback_token(0), Some(true));
        assert_eq!(d.ends_with_rollback_token(1), Some(false));
        assert_eq!(d.ends_with_rollback_token(2), None);
        assert_eq!(
            d.process_token_rows(&[vec![0], vec![]]),
            Err(Error::InvalidTokenRow)
        );
        assert_eq!(d.ends_with_rollback_token(0), Some(true));
        d.process_tokens(&[0, 7]).unwrap();
        assert_eq!(d.ends_with_rollback, vec![false, true]);
        d.reset_batch(None);
        assert_eq!(d.ends_with_rollback, vec![false, false]);
        d.process_tokens(&[7, 7]).unwrap();
        assert_eq!(d.ends_with_rollback, vec![true, true]);
    }

    #[test]
    fn rollback_last_simultaneous_match_controls_flag() {
        for (sequences, expected) in [
            (vec![vec![7], vec![1, 7]], false),
            (vec![vec![1, 7], vec![7]], true),
        ] {
            let mut d = detector(1, &[]);
            for sequence in sequences {
                d.add_stop_sequence(&sequence).unwrap();
            }
            d.mark_for_rollback(7);
            d.process_tokens(&[1]).unwrap();
            d.process_tokens(&[7]).unwrap();
            assert_eq!(d.ends_with_rollback_token(0), Some(expected));
        }
    }

    #[test]
    fn marking_is_not_retroactive_and_duplicates_retain_mark() {
        let mut d = detector(1, &[&[7]]);
        d.process_tokens(&[7]).unwrap();
        assert!(d.mark_for_rollback(7));
        assert_eq!(d.ends_with_rollback_token(0), Some(false));
        d.add_stop_sequence(&[7]).unwrap();
        d.reset_batch(None);
        d.process_tokens(&[7]).unwrap();
        assert_eq!(d.ends_with_rollback_token(0), Some(true));
    }

    #[test]
    fn nested_rows_match_flat_processing() {
        let mut nested = detector(2, &[&[1, 2], &[2]]);
        let mut flat = detector(2, &[&[1, 2], &[2]]);
        for tokens in [[1, 0], [2, 1], [3, 2], [4, 5]] {
            nested
                .process_token_rows(&tokens.map(|token| vec![token]))
                .unwrap();
            flat.process_tokens(&tokens).unwrap();
            assert_eq!(nested.found, flat.found);
            assert_eq!(nested.discard, flat.discard);
            assert_eq!(nested.partial, flat.partial);
            assert_eq!(nested.progress, flat.progress);
        }
    }

    #[test]
    fn nested_invalid_later_row_is_atomic_even_after_stop() {
        let mut d = detector(2, &[&[1, 2]]);
        d.process_tokens(&[1, 1]).unwrap();
        for rows in [vec![vec![2], vec![]], vec![vec![2], vec![2, 3]]] {
            assert_eq!(d.process_token_rows(&rows), Err(Error::InvalidTokenRow));
            assert_eq!(d.partial, vec![1, 1]);
            assert_eq!(d.discard, vec![0, 0]);
            assert_eq!(d.progress, vec![vec![1], vec![1]]);
            assert_eq!(d.found, vec![false, false]);
        }
        d.process_token_rows(&[vec![2], vec![2]]).unwrap();
        assert_eq!(
            d.process_token_rows(&[vec![7], vec![]]),
            Err(Error::InvalidTokenRow)
        );
        assert_eq!(d.discard, vec![2, 2]);
    }

    #[test]
    fn nested_validation_preserves_native_error_precedence() {
        let mut d = detector(2, &[]);
        assert_eq!(
            d.process_token_rows(&[vec![]]),
            Err(Error::BatchSizeMismatch)
        );
        assert_eq!(
            d.process_token_rows(&[vec![1], vec![]]),
            Err(Error::InvalidTokenRow)
        );
        assert_eq!(
            d.process_token_rows(&[vec![1], vec![2]]),
            Err(Error::NoStopSequences)
        );
        assert_eq!(d.partial, vec![0, 0]);
        assert_eq!(d.discard, vec![0, 0]);
        assert_eq!(d.found, vec![false, false]);
    }

    #[test]
    fn batching_counts_tokens_to_discard_after_early_stop() {
        let mut d = detector(2, &[&[5], &[7, 8, 9]]);
        for pair in [[3, 1], [6, 0], [7, 0], [8, 0], [9, 0], [10, 6], [11, 5]] {
            d.process_tokens(&pair).unwrap();
        }
        assert!(d.all_done());
        assert_eq!(d.steps_before_stop_tokens(), &[5, 1]);
        assert_eq!(d.max_partial_stop_token_length(0), Some(3));
    }

    #[test]
    fn reset_retains_sequences_and_supports_repeat_generation() {
        let mut d = detector(1, &[&[1, 2]]);
        for _ in 0..100 {
            d.process_tokens(&[1]).unwrap();
            d.process_tokens(&[2]).unwrap();
            assert!(d.all_done());
            d.reset_batch(None);
            assert!(!d.all_done());
            assert_eq!(d.steps_before_stop_tokens(), &[0]);
        }
        d.reset_batch(NonZeroUsize::new(2));
        d.process_tokens(&[1, 1]).unwrap();
        d.process_tokens(&[2, 2]).unwrap();
        assert_eq!(d.stop_tokens_found(), &[true, true]);
    }

    #[test]
    fn invalid_inputs_leave_state_unchanged() {
        let mut d = detector(1, &[]);
        assert_eq!(d.process_tokens(&[1]), Err(Error::NoStopSequences));
        assert_eq!(d.add_stop_sequence(&[]), Err(Error::EmptyStopSequence));
        d.add_stop_sequence(&[1, 2]).unwrap();
        d.add_stop_sequence(&[1, 2]).unwrap();
        d.process_tokens(&[1]).unwrap();
        assert_eq!(d.process_tokens(&[]), Err(Error::BatchSizeMismatch));
        assert_eq!(d.max_partial_stop_token_length(0), Some(1));
        assert_eq!(d.max_partial_stop_token_length(1), None);
    }

    #[test]
    fn preserves_upstream_overlap_restart_behavior() {
        let mut d = detector(1, &[&[1, 1, 2]]);
        for token in [1, 1, 1, 2] {
            d.process_tokens(&[token]).unwrap();
        }
        // Upstream restarts at length 1 rather than finding the longest suffix.
        assert!(!d.all_done());
        assert_eq!(d.max_partial_stop_token_length(0), Some(0));
    }

    #[test]
    fn last_simultaneous_match_controls_discard_count() {
        let mut d = detector(1, &[&[1, 2], &[2]]);
        d.process_tokens(&[1]).unwrap();
        d.process_tokens(&[2]).unwrap();
        assert_eq!(d.steps_before_stop_tokens(), &[1]);
        assert_eq!(d.max_partial_stop_token_length(0), Some(2));
    }
}
