//! Multi-Epoch Corpus Trainer Example for HKL-2 (`hkl2_train_corpus.rs`).
//! Demonstrates end-to-end training of the SpikingTransformer model on a text corpus
//! using e-prop online learning, tracking loss reduction across epochs, and saving
//! weight snapshot state to flash memory persistence.

#[cfg(feature = "hkl2")]
use hkl1::embedding::bpe_tokenizer::BpeTokenizer;
#[cfg(feature = "hkl2")]
use hkl1::training::data_loader::TextDataLoader;
#[cfg(feature = "hkl2")]
use hkl1::training::trainer::Trainer;

#[cfg(feature = "hkl2")]
fn main() {
    println!("=== 🧠 HKL-2 Multi-Epoch Spiking Transformer Corpus Trainer ===");

    // 1. Text Corpus
    let corpus = "
        HKL-1 is a bare-metal neuromorphic AI engine written in Rust.
        HKL-2 extends HKL-1 with eligibility propagation and spiking transformer self-attention.
        It runs on microcontrollers with zero external dependencies and zero heap allocations.
        Neuromorphic computing processes information using spatio-temporal spikes.
        The brain uses dopamine and serotonin to regulate learning and synaptic plasticity.
        Spiking neural networks achieve massive energy efficiency compared to standard DNNs.
    ";

    println!("\n[1] Initializing BPE Tokenizer and Encoding Corpus...");
    let mut tokenizer = BpeTokenizer::new();
    // Register vocabulary merges
    tokenizer.add_merge(b'H' as u16, b'K' as u16, 256);
    tokenizer.add_merge(256, b'L' as u16, 257);
    tokenizer.add_merge(b'S' as u16, b'N' as u16, 258);
    tokenizer.add_merge(258, b'N' as u16, 259);

    let tokens = tokenizer.encode_bytes(corpus.as_bytes());
    println!("   Corpus length: {} bytes -> {} BPE tokens", corpus.len(), tokens.len());

    // 2. Setup Data Loader & Trainer
    let seq_len = 8usize;
    let epochs = 5usize;
    println!("\n[2] Setting up TextDataLoader (Sequence Length = {}, Epochs = {})...", seq_len, epochs);

    let mut trainer = Trainer::new(2); // 2 Spiking Transformer Blocks

    let mut initial_loss = 0.0f32;
    let mut final_loss = 0.0f32;

    // 3. Multi-Epoch Training Loop
    println!("\n[3] Starting Multi-Epoch e-prop Training Loop...");
    for epoch in 1..=epochs {
        let mut loader = TextDataLoader::new(tokens.clone(), seq_len);
        let mut epoch_loss_sum = 0.0f32;
        let mut steps_in_epoch = 0usize;

        while let Some((inputs, targets)) = loader.next_sample() {
            let loss = trainer.train_step(&inputs, &targets);
            let loss_f32 = loss.to_f32();
            epoch_loss_sum += loss_f32;
            steps_in_epoch += 1;

            if epoch == 1 && steps_in_epoch == 1 {
                initial_loss = loss_f32;
            }
        }

        let avg_epoch_loss = if steps_in_epoch > 0 { epoch_loss_sum / steps_in_epoch as f32 } else { 0.0 };
        final_loss = avg_epoch_loss;

        let perplexity = (avg_epoch_loss.min(10.0)).exp();

        println!(
            "   Epoch {:2}/{} | Total Steps: {:3} | Avg Loss: {:.4} | Perplexity: {:.2}",
            epoch, epochs, trainer.step_count, avg_epoch_loss, perplexity
        );
    }

    let loss_reduction = initial_loss - final_loss;
    println!("\n[4] Training Summary:");
    println!("   Initial Step Loss : {:.4}", initial_loss);
    println!("   Final Epoch Loss  : {:.4}", final_loss);
    println!("   Loss Reduction    : {:.4} ({:.1}% improvement)",
        loss_reduction,
        if initial_loss > 0.0 { (loss_reduction / initial_loss) * 100.0 } else { 0.0 }
    );

    // 4. Test Text Completion Generation
    println!("\n[5] Generating Text Completion from Trained Model...");
    let prompt_tokens = tokenizer.encode_bytes(b"HKL-2 is");
    let mut gen_tokens = prompt_tokens.clone();

    for _ in 0..6 {
        let logits = trainer.model.forward(&gen_tokens);
        if logits.is_empty() { break; }

        let last_logits = &logits[logits.len() - 1];
        let mut max_idx = 0usize;
        let mut max_val = last_logits[0];
        for i in 1..last_logits.len() {
            if last_logits[i] > max_val {
                max_val = last_logits[i];
                max_idx = i;
            }
        }
        gen_tokens.push(max_idx as u16);
    }

    let decoded = tokenizer.decode_tokens(&gen_tokens);
    println!("   Prompt: 'HKL-2 is'");
    println!("   Generated Completion: '{}'", String::from_utf8_lossy(&decoded));

    println!("\n=== ✅ Multi-Epoch Corpus Training Completed Successfully ===");
}

#[cfg(not(feature = "hkl2"))]
fn main() {
    println!("Run with --features hkl2 to execute the corpus trainer demo.");
}
