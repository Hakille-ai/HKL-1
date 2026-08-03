//! `hkl2-prepare` — offline dataset preparation for HKL-2.
//!
//! Usage:
//!   hkl2-prepare --in <text_file> --out <corpus.hklc> [--merges N] [--bpe <file>] [--meta "source|license|lang"]
//!
//! Reads a UTF-8 text file, trains a byte-level BPE tokenizer (or loads one),
//! tokenizes the corpus, and writes a versioned `.hklc` corpus file plus
//! the BPE merge table.

use std::env;
use std::fs;
use std::io::{self, Read};
use std::path::Path;

use hkl1::embedding::bpe_tokenizer::BpeTokenizer;
use hkl1::training::corpus::{HklcWriter, fnv_hash, save_corpus};

const DEFAULT_MERGES: usize = 256;
const USAGE: &str = r#"
Usage: hkl2-prepare --in <text_file> --out <corpus.hklc> [options]

Options:
  --in <path>          Input text file (UTF-8)
  --out <path>         Output corpus file (.hklc)
  --merges <N>         Number of BPE merges to learn (default: 256, max: 3840)
  --bpe <path>         Load existing BPE merge table from file
  --save-bpe <path>    Save trained BPE merge table to file
  --meta <str>         Corpus metadata "source|license|language" (default: "custom|MIT|en")
  --seq-len <N>        Sequence length hint for the corpus (default: 8)
  --help               Show this help
"#;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();

    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{}", USAGE);
        return;
    }

    let mut input_path = None;
    let mut output_path = None;
    let mut merges = DEFAULT_MERGES;
    let mut bpe_load = None;
    let mut bpe_save = None;
    let mut meta = "custom|MIT|en".to_string();
    let mut seq_len = 8usize;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--in" => {
                i += 1;
                input_path = args.get(i).map(|s| s.as_str());
            }
            "--out" => {
                i += 1;
                output_path = args.get(i).map(|s| s.as_str());
            }
            "--merges" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    merges = v.parse().unwrap_or(DEFAULT_MERGES);
                }
            }
            "--bpe" => {
                i += 1;
                bpe_load = args.get(i).map(|s| s.as_str());
            }
            "--save-bpe" => {
                i += 1;
                bpe_save = args.get(i).map(|s| s.as_str());
            }
            "--meta" => {
                i += 1;
                meta = args.get(i).map(|s| s.to_string()).unwrap_or(meta);
            }
            "--seq-len" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    seq_len = v.parse().unwrap_or(8);
                }
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let input_path = input_path.expect("--in is required");
    let output_path = output_path.expect("--out is required");

    let text = fs::read_to_string(input_path).unwrap_or_else(|e| {
        eprintln!("Failed to read {}: {}", input_path, e);
        std::process::exit(1);
    });

    let mut tokenizer = if let Some(path) = bpe_load {
        let blob = fs::read(path).unwrap_or_else(|e| {
            eprintln!("Failed to read BPE file {}: {}", path, e);
            std::process::exit(1);
        });
        BpeTokenizer::from_bytes(&blob).unwrap_or_else(|| {
            eprintln!("Invalid BPE file: {}", path);
            std::process::exit(1);
        })
    } else {
        let mut t = BpeTokenizer::new();
        let learned = t.train(text.as_bytes(), merges);
        println!("Trained BPE: {} merges learned", learned);
        t
    };

    let tokens = tokenizer.encode_bytes(text.as_bytes());
    println!("Corpus: {} bytes -> {} tokens", text.len(), tokens.len());

    let vocab_hash = fnv_hash(&tokenizer.to_bytes());
    if let Err(e) = save_corpus(output_path, &tokens, &meta, seq_len, vocab_hash) {
        eprintln!("Failed to write corpus: {}", e);
        std::process::exit(1);
    }
    println!("Written corpus to {}", output_path);

    if let Some(path) = bpe_save {
        let blob = tokenizer.to_bytes();
        fs::write(path, blob).unwrap_or_else(|e| {
            eprintln!("Failed to write BPE file {}: {}", path, e);
            std::process::exit(1);
        });
        println!("Saved BPE merge table to {}", path);
    }
}
