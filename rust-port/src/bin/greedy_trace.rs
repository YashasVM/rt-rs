// Copyright 2026 rt-rs contributors.
// SPDX-License-Identifier: Apache-2.0
use rt_rs_stop::greedy::greedy_token_ids;
use std::io::{self, BufRead};
use std::num::NonZeroUsize;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 6 {
        assert_eq!(args[1], "reduction");
        let batch = NonZeroUsize::new(args[2].parse().unwrap()).unwrap();
        let sequence = NonZeroUsize::new(args[3].parse().unwrap()).unwrap();
        let vocab: usize = args[4].parse().unwrap();
        let rounds: usize = args[5].parse().unwrap();
        assert!(vocab > 0 && rounds > 0);
        let count = batch
            .get()
            .checked_mul(sequence.get())
            .unwrap()
            .checked_mul(vocab)
            .unwrap();
        let mut logits: Vec<f32> = (0..count).map(|i| (i % 101) as f32 - 50.0).collect();
        let mut checksum = 0usize;
        let start = std::time::Instant::now();
        for repeat in 0..rounds {
            logits[0] = (repeat % 7) as f32 - 3.0;
            let ids = greedy_token_ids(std::hint::black_box(&logits), batch, sequence).unwrap();
            for row in ids {
                for id in row {
                    checksum += std::hint::black_box(id);
                }
            }
        }
        println!("{} {}", start.elapsed().as_nanos(), checksum);
        return;
    }

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
