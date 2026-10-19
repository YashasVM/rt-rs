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
}

/// Incremental token-ID stop detection with LiteRT-LM v0.18.0 semantics.
///
/// Each call consumes exactly one token per batch item. This owns all state
/// and does not call native code. Counts use usize instead of C++ signed int.
#[derive(Debug)]
pub struct StopTokenDetector {
    sequences: Vec<Vec<i32>>,
    progress: Vec<Vec<usize>>,
    partial: Vec<usize>,
    found: Vec<bool>,
    discard: Vec<usize>,
}

impl StopTokenDetector {
    pub fn new(batch_size: NonZeroUsize) -> Self {
        let mut detector = Self {
            sequences: Vec::new(),
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
            for row in &mut self.progress {
                row.push(0);
            }
        }
        Ok(())
    }

    /// Retains stop sequences; None retains the existing nonzero batch size.
    pub fn reset_batch(&mut self, batch_size: Option<NonZeroUsize>) {
        let size = batch_size.map_or(self.found.len(), NonZeroUsize::get);
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
        if self.sequences.is_empty() {
            return Err(Error::NoStopSequences);
        }
        let batch = tokens
            .iter()
            .zip(&mut self.found)
            .zip(&mut self.discard)
            .zip(&mut self.partial)
            .zip(&mut self.progress);
        for ((((&token, found), discard), partial), row) in batch {
            if *found {
                *discard += 1;
                continue;
            }
            let mut maximum = 0;
            for (progress, sequence) in row.iter_mut().zip(&self.sequences) {
                if sequence.get(*progress) == Some(&token) {
                    *progress += 1;
                } else {
                    *progress = usize::from(sequence.first() == Some(&token));
                }
                if *progress > 0 && *progress == sequence.len() {
                    *found = true;
                    *discard = sequence.len();
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
