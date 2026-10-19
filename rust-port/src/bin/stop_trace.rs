// SPDX-License-Identifier: Apache-2.0
use rt_rs_stop::{Error, StopTokenDetector};
use std::io::{self, BufRead};
use std::num::NonZeroUsize;
use std::time::Instant;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("bench") {
        let nested = args[2].starts_with("nested-");
        let mode = args[2].strip_prefix("nested-").unwrap_or(&args[2]);
        let mut rows = vec![vec![0]; 4];
        let rounds: usize = args[3].parse().unwrap();
        let mut d = StopTokenDetector::new(NonZeroUsize::new(4).unwrap());
        for sequence in [&[7, 8, 9][..], &[4, 4, 5], &[99]] {
            d.add_stop_sequence(sequence).unwrap();
        }
        let mut checksum = 0usize;
        let start = Instant::now();
        for _ in 0..rounds {
            d.reset_batch(None);
            for step in 0..64 {
                let mut tokens = [0; 4];
                for (i, token) in tokens.iter_mut().enumerate() {
                    *token = match mode {
                        "no-match" => ((step + i) % 4) as i32,
                        "partial" => [7, 8, 0][(step + i) % 3],
                        "early-stop" => ((step + i) % 13) as i32,
                        _ => panic!("unknown benchmark mode"),
                    };
                }
                if nested {
                    for (row, token) in rows.iter_mut().zip(tokens) {
                        row[0] = token;
                    }
                    d.process_token_rows(std::hint::black_box(&rows)).unwrap();
                } else {
                    d.process_tokens(std::hint::black_box(&tokens)).unwrap();
                }
                checksum += d.steps_before_stop_tokens().iter().sum::<usize>();
                checksum += d.max_partial_stop_token_length(0).unwrap();
            }
        }
        println!("{} {}", start.elapsed().as_nanos(), checksum);
        return;
    }
    let current_main = args.get(1).map(String::as_str) == Some("--current-main");
    let mut d = StopTokenDetector::new(NonZeroUsize::new(1).unwrap());
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let mut words = line.split_whitespace();
        let command = words.next().unwrap();
        let values: Vec<i32> = words.map(|v| v.parse().unwrap()).collect();
        let status = match command {
            "N" => {
                d = StopTokenDetector::new(NonZeroUsize::new(values[0] as usize).unwrap());
                0
            }
            "A" => usize::from(d.add_stop_sequence(&values).is_err()) * 3,
            "R" => {
                d.reset_batch(NonZeroUsize::new(values[0] as usize));
                0
            }
            "P" => usize::from(d.process_tokens(&values).is_err()) * 3,
            "M" if current_main => usize::from(!d.mark_for_rollback(values[0])) * 5,
            "V" => {
                let mut rows = Vec::new();
                let mut offset = 0;
                while offset < values.len() {
                    let length = values[offset] as usize;
                    offset += 1;
                    rows.push(values[offset..offset + length].to_vec());
                    offset += length;
                }
                match d.process_token_rows(&rows) {
                    Ok(()) => 0,
                    Err(Error::InvalidTokenRow) => 13,
                    Err(_) => 3,
                }
            }
            _ => panic!("unknown trace command"),
        };
        print!("{} {}", status, usize::from(d.all_done()));
        for i in 0..d.stop_tokens_found().len() {
            print!(
                " {} {} {}",
                usize::from(d.stop_tokens_found()[i]),
                d.steps_before_stop_tokens()[i],
                d.max_partial_stop_token_length(i).unwrap()
            );
            if current_main {
                print!(" {}", usize::from(d.ends_with_rollback_token(i).unwrap()));
            }
        }
        println!();
    }
}
