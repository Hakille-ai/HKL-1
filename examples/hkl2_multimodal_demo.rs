//! HKL-2 Multi-Modal Spiking Foundation Model Demo.
//! Demonstrates unmediated processing of Text, Audio (Cochlea v2),
//! and Vision (Retina v2) through the Spiking Transformer backbone.

#[cfg(feature = "hkl2")]
use hkl1::embedding::bpe_tokenizer::BpeTokenizer;
#[cfg(feature = "hkl2")]
use hkl1::encoders::audio_encoder::AudioSpikeEncoder;
#[cfg(feature = "hkl2")]
use hkl1::encoders::vision_encoder::VisionSpikeEncoder;
#[cfg(feature = "hkl2")]
use hkl1::training::data_loader::TextDataLoader;
#[cfg(feature = "hkl2")]
use hkl1::training::trainer::Trainer;
#[cfg(feature = "hkl2")]
use hkl1::vision::retina::VISION_PIXELS;

#[cfg(feature = "hkl2")]
fn main() {
    println!("=== 🧠 HKL-2 Multi-Modal Spiking Foundation Model Demo ===");

    // 1. Text Pipeline & BPE Tokenization
    println!("\n[1] Text Tokenization & DataLoader...");
    let mut tokenizer = BpeTokenizer::new();
    tokenizer.add_merge(b'h' as u16, b'e' as u16, 256);
    tokenizer.add_merge(256, b'l' as u16, 257);
    tokenizer.add_merge(257, b'l' as u16, 258);
    tokenizer.add_merge(258, b'o' as u16, 259);

    let corpus_text = b"hello world hello hkl2 foundation model";
    let tokens = tokenizer.encode_bytes(corpus_text);
    println!("   Text: {:?}", core::str::from_utf8(corpus_text).unwrap());
    println!("   Tokens: {:?}", tokens);

    let mut data_loader = TextDataLoader::new(tokens, 4);

    // 2. Trainer Initialization & e-prop Training
    println!("\n[2] Spiking Transformer & e-prop Learning...");
    let mut trainer = Trainer::new(2); // 2-layer Spiking Transformer

    let mut epoch_loss = hkl1::core::math::FixedPoint::ZERO;
    let mut step_count = 0;
    while let Some((inputs, targets)) = data_loader.next_sample() {
        let loss = trainer.train_step(&inputs, &targets);
        epoch_loss += loss;
        step_count += 1;
        println!("   Step {:02}: Loss = {:.4}", step_count, loss.to_f32());
    }

    // 3. Audio Encoder Pipeline (PCM 16kHz -> 256D Spike Matrix)
    println!("\n[3] Audio Encoder Pipeline (Cochlea v2 -> 256D Spikes)...");
    let mut audio_encoder = AudioSpikeEncoder::new();
    audio_encoder.init_random(42);

    let mut pcm_sample = [0i16; 512];
    for (i, sample) in pcm_sample.iter_mut().enumerate() {
        let t = i as f32 / 16000.0;
        *sample = ((2.0 * core::f32::consts::PI * 440.0 * t).sin() * 16000.0) as i16;
    }
    let audio_spikes = audio_encoder.encode_pcm(&pcm_sample, 100);
    let total_audio_spikes = audio_spikes
        .iter()
        .flatten()
        .filter(|spike| **spike)
        .count();
    println!("   440 Hz Sine Tone -> Cochlea Gammatone 32-band -> 256D Spikes");
    println!(
        "   Generated {} audio spikes across 4 timesteps",
        total_audio_spikes
    );

    // 4. Vision Encoder Pipeline (Retina 32x32 -> 256D Spike Matrix)
    println!("\n[4] Vision Encoder Pipeline (Retina DoG -> 256D Spikes)...");
    let mut vision_encoder = VisionSpikeEncoder::new();
    vision_encoder.init_random(123);

    let mut video_frame = [0u8; VISION_PIXELS];
    for (y, row) in video_frame.chunks_mut(32).enumerate() {
        for (x, pixel) in row.iter_mut().enumerate() {
            *pixel = if (x + y) % 2 == 0 { 255 } else { 0 };
        }
    }
    let vision_spikes = vision_encoder.encode_frame(&video_frame, 200);
    let total_vision_spikes = vision_spikes
        .iter()
        .flatten()
        .filter(|spike| **spike)
        .count();
    println!("   32x32 Checkerboard Frame -> Retinal DoG 1024-px -> 256D Spikes");
    println!(
        "   Generated {} vision spikes across 4 timesteps",
        total_vision_spikes
    );

    println!("\n=== ✅ Multi-Modal Pipeline Execution Complete ===");
}

#[cfg(not(feature = "hkl2"))]
fn main() {
    println!("Run with --features hkl2 to execute the multi-modal foundation model demo.");
}
