// Copyright 2026 rt-rs contributors.
// SPDX-License-Identifier: Apache-2.0
use rt_rs_stop::half::{greedy_fp16_token_ids, half_to_f32};
use std::io::{self, BufRead};
use std::num::NonZeroUsize;
fn main() {
    if std::env::args().nth(1).as_deref() == Some("convert") {
        for value in 0..=u16::MAX {
            println!("{:08x}", half_to_f32(value).to_bits());
        }
        return;
    }
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 7 {
        let mode = &args[1];
        assert!(matches!(mode.as_str(), "fused" | "fused-copy" | "staged"));
        let batch = NonZeroUsize::new(args[2].parse().unwrap()).unwrap();
        let sequence = NonZeroUsize::new(args[3].parse().unwrap()).unwrap();
        let vocab: usize = args[4].parse().unwrap();
        let rounds: usize = args[5].parse().unwrap();
        assert!(vocab > 1 && rounds > 0);
        let count = batch
            .get()
            .checked_mul(sequence.get())
            .unwrap()
            .checked_mul(vocab)
            .unwrap();
        let mut state = 1800u32;
        let mut values: Vec<u16> = (0..count)
            .map(|i| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                match args[6].as_str() {
                    "random" => {
                        let mut bits = (state >> 16) as u16;
                        if bits & 0x7fff >= 0x7c00 {
                            bits &= !0x0400;
                        }
                        bits
                    }
                    "ascending" => ((i % vocab) * 0x7bff / (vocab - 1)) as u16,
                    "periodic" => ((i % 101) * 0x100) as u16,
                    _ => panic!("unknown input pattern"),
                }
            })
            .collect();
        let mut converted = Vec::new();
        let mut checksum = 0usize;
        let start = std::time::Instant::now();
        for repeat in 0..rounds {
            values[0] = if repeat % 2 == 0 { 0x3c00 } else { 0xbc00 };
            let ids = match mode.as_str() {
                "staged" => {
                    converted.resize(count, 0.0);
                    for (slot, &bits) in converted.iter_mut().zip(std::hint::black_box(&values)) {
                        *slot = half_to_f32(bits);
                    }
                    rt_rs_stop::greedy::greedy_token_ids(&converted, batch, sequence).unwrap()
                }
                "fused-copy" => {
                    let staging: Vec<u16> = std::hint::black_box(&values).clone();
                    greedy_fp16_token_ids(std::hint::black_box(&staging), batch, sequence).unwrap()
                }
                _ => greedy_fp16_token_ids(std::hint::black_box(&values), batch, sequence).unwrap(),
            };
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
        let batch: usize = words.next().unwrap().parse().unwrap();
        let sequence: usize = words.next().unwrap().parse().unwrap();
        let count: usize = words.next().unwrap().parse().unwrap();
        let values: Vec<_> = words.map(|s| u16::from_str_radix(s, 16).unwrap()).collect();
        assert_eq!(count, values.len());
        let (Some(batch), Some(sequence)) = (NonZeroUsize::new(batch), NonZeroUsize::new(sequence)) else {
            println!("ERR"); continue;
        };
        match greedy_fp16_token_ids(&values, batch, sequence) {
            Ok(rows) => {
                print!("OK");
                for id in rows.into_iter().flatten() {
                    print!(" {id}");
                }
                println!();
            }
            Err(_) => println!("ERR"),
        }
    }
}
