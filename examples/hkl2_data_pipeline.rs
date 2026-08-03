//! Full Data Pipeline Example for HKL-2 (`hkl2_data_pipeline.rs`).
//! Demonstrates the complete offline training workflow:
//!   1. Load prepared corpus (.hklc) and BPE merge table
//!   2. Split into train/validation
//!   3. Train with batched e-prop + metacognitive tuning
//!   4. Evaluate on held-out split
//!   5. Save best checkpoint
//!   6. Generate text completion from trained model

#[cfg(feature = "hkl2")]
use hkl1::embedding::bpe_tokenizer::BpeTokenizer;
#[cfg(feature = "hkl2")]
use hkl1::training::checkpoint::load_checkpoint;
#[cfg(feature = "hkl2")]
use hkl1::training::corpus::load_corpus;
#[cfg(feature = "hkl2")]
use hkl1::training::data_loader::TextDataLoader;
#[cfg(feature = "hkl2")]
use hkl1::training::dataset::split_tokens;
#[cfg(feature = "hkl2")]
use hkl1::training::run::TrainingRun;

#[cfg(feature = "hkl2")]
fn main() {
    println!("=== 🧠 HKL-2 Full Data Pipeline Demo ===");

    // 1. Load corpus and BPE
    println!("\n[1] Loading corpus and tokenizer...");
    let corpus = load_corpus("data/demo_corpus.hklc").expect("failed to load corpus");
    let tokenizer = BpeTokenizer::from_bytes(&std::fs::read("data/demo.bpe").expect("read bpe"))
        .expect("failed to load BPE");

    println!(
        "   Corpus: {} tokens, vocab hash: {:x}",
        corpus.token_count(),
        corpus.vocab_hash
    );
    println!("   BPE merges: {}", tokenizer.merge_count);

    // 2. Split train/val
    println!("\n[2] Splitting train/validation (90/10)...");
    let (train_tokens, val_tokens) = split_tokens(&corpus.tokens(), corpus.seq_len.max(8), 0.1, 42);
    println!(
        "   Train: {} tokens, Val: {} tokens",
        train_tokens.len(),
        val_tokens.len()
    );

    // 3. Initialize training run
    println!(
        "\n[3] Initializing TrainingRun (2 layers, seq_len={})....",
        corpus.seq_len
    );
    let mut runner = TrainingRun::new(2);
    runner.tokenizer = Some(tokenizer.clone());
    runner.checkpoint_dir = Some("data/snapshots".to_string());

    // 4. Training loop
    println!("\n[4] Starting training (3 epochs)...");
    let mut train_loader = TextDataLoader::new(train_tokens, corpus.seq_len.max(8));
    let mut val_loader = TextDataLoader::new(val_tokens, corpus.seq_len.max(8));

    for epoch in 1..=3 {
        let report = runner.run_epoch(&mut train_loader, Some(&mut val_loader), 8);
        println!(
            "   Epoch {}/3 | Steps: {} | Avg Loss: {:.4} | Action: {} | LR Scale: {:.3} | Val Loss: {:.4} | Val Acc: {:.2}%",
            epoch,
            report.steps,
            report.avg_loss,
            report.tuning_action,
            report.learning_scale,
            report.eval.loss,
            report.eval.accuracy * 100.0
        );
    }

    // 5. Checkpoint already saved via runner (best val loss)
    println!("\n[5] Best checkpoint saved to data/snapshots/slot_*.hklk");

    // 6. Load best checkpoint and generate
    println!("\n[6] Loading best checkpoint and generating text...");
    // Runner saves to slot_{epoch % 3} when improved; epoch 1 -> slot_1
    let best_path = "data/snapshots/slot_1.hklk";
    let (mut model, gen_tokenizer) = if let Ok(ckpt) = load_checkpoint(best_path) {
        println!("   Loaded checkpoint (step {})", ckpt.step_count);
        (ckpt.model, ckpt.tokenizer)
    } else {
        println!(
            "   (Checkpoint not found at {}, using final runner state)",
            best_path
        );
        (runner.trainer.model, tokenizer)
    };

    let prompt = b"HKL-2 is";
    let prompt_tokens = gen_tokenizer.encode_bytes(prompt);
    let mut gen_tokens = prompt_tokens.clone();

    for _ in 0..16 {
        let logits = model.forward(&gen_tokens);
        if logits.is_empty() {
            break;
        }
        let last = &logits[logits.len() - 1];
        let (mut best_idx, mut best_val) = (0usize, last[0]);
        for (i, &v) in last.iter().enumerate().skip(1) {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        gen_tokens.push(best_idx as u16);
        if best_idx == 0 || best_idx == 10 {
            break;
        }
    }

    let decoded = gen_tokenizer.decode_tokens(&gen_tokens);
    println!("   Prompt: '{}'", String::from_utf8_lossy(prompt));
    println!("   Generated: '{}'", String::from_utf8_lossy(&decoded));

    println!("\n=== ✅ Full Data Pipeline Completed Successfully ===");
}

#[cfg(not(feature = "hkl2"))]
fn main() {
    println!("Run with --features hkl2 to execute the data pipeline demo.");
}
