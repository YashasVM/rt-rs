// Copyright 2026 rt-rs contributors.
// SPDX-License-Identifier: Apache-2.0
use rt_rs_stop::greedy::greedy_token_ids;
use std::io::{self, BufRead};
use std::num::NonZeroUsize;

fn main() {
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let mut words = line.split_whitespace();
        let batch = NonZeroUsize::new(words.next().unwrap().parse().unwrap()).unwrap();
        let sequence = NonZeroUsize::new(words.next().unwrap().parse().unwrap()).unwrap();
        let count: usize = words.next().unwrap().parse().unwrap();
        let logits: Vec<_> = words
            .map(|s| f32::from_bits(u32::from_str_radix(s, 16).unwrap()))
            .collect();
        assert_eq!(count, logits.len());
        match greedy_token_ids(&logits, batch, sequence) {
            Ok(rows) => {
                print!("OK");
                for token in rows.into_iter().flatten() {
                    print!(" {token}");
                }
                println!();
            }
            Err(_) => println!("ERR"),
        }
    }
}
