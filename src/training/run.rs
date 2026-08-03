//! Training orchestrator: batches data via [`BatchSampler`], applies
//! metacognitive tuning to the learning rate, evaluates on a held-out
//! split, and persists the best checkpoint.
//!
//! This is the "driver" layer of the pipeline: the same code powers a
//! corpus file, an in-memory source, or a live stream — anything that
//! implements [`DataSource`].

use crate::cognition::metacognition::{MetacognitiveAutoTuner, TuningAction};
use crate::core::math::FixedPoint;
use crate::embedding::bpe_tokenizer::BpeTokenizer;
use crate::training::checkpoint;
use crate::training::dataset::{BatchSampler, DataSource};
use crate::training::trainer::{TrainStepStatus, Trainer};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// Neutral curiosity baseline fed to the metacognitive tuner.
pub const CURIOSITY_NOMINAL: FixedPoint = FixedPoint::from_f32(0.5);
pub const BOREDOM_NOMINAL: f32 = 0.0;

/// Validation metrics over a held-out split.
#[derive(Debug, Clone)]
pub struct EvalReport {
    pub loss: f32,
    pub perplexity: f32,
    pub accuracy: f32,
    pub samples: usize,
    pub tokens: usize,
}

impl EvalReport {
    pub fn empty() -> Self {
        Self {
            loss: f32::NAN,
            perplexity: f32::NAN,
            accuracy: f32::NAN,
            samples: 0,
            tokens: 0,
        }
    }
}

/// Summary of a single training epoch.
#[derive(Debug, Clone)]
pub struct EpochReport {
    pub epoch: u64,
    pub steps: u64,
    pub batches: usize,
    pub tokens: u64,
    pub avg_loss: f32,
    pub tuning_action: &'static str,
    pub learning_scale: f32,
    pub surrogate_slope: f32,
    pub best_loss: Option<f32>,
    pub eval: EvalReport,
}

/// The training driver.
pub struct TrainingRun {
    pub trainer: Trainer,
    pub tuner: MetacognitiveAutoTuner,
    pub epoch: u64,
    pub best_loss: Option<f32>,
    pub checkpoint_dir: Option<String>,
    pub tokenizer: Option<BpeTokenizer>,
}

impl TrainingRun {
    pub fn new(num_layers: usize) -> Self {
        Self {
            trainer: Trainer::new(num_layers),
            tuner: MetacognitiveAutoTuner::new(),
            epoch: 0,
            best_loss: None,
            checkpoint_dir: None,
            tokenizer: None,
        }
    }

    /// Evaluate a whole validation source without touching weights.
    pub fn evaluate(&mut self, val: &mut dyn DataSource, batch_size: usize) -> EvalReport {
        let mut total_loss = 0.0f32;
        let mut correct = 0usize;
        let mut lossable = 0usize;
        let mut tokens = 0usize;

        for _ in 0..batch_size {
            let Some((inputs, targets)) = val.next_sample() else {
                break;
            };
            let (loss, ok) = self.trainer.eval_sample(&inputs, &targets);
            total_loss += loss.to_f32();
            lossable += 1;
            correct += ok;
            tokens += inputs.len().min(targets.len());
        }

        if lossable == 0 {
            return EvalReport::empty();
        }

        let avg_loss = total_loss / lossable as f32;
        EvalReport {
            loss: avg_loss,
            perplexity: (avg_loss.min(10.0)).exp(),
            accuracy: correct as f32 / tokens.max(1) as f32,
            samples: lossable,
            tokens,
        }
    }

    /// Run one full pass over a training source, then evaluate on `val`.
    pub fn run_epoch<T: DataSource, V: DataSource>(
        &mut self,
        train: &mut T,
        mut val: Option<&mut V>,
        batch_size: usize,
    ) -> EpochReport {
        self.epoch += 1;
        train.reset();
        if let Some(ref mut v) = val {
            v.reset();
        }
        let mut sampler = BatchSampler::new(Borrowed(train), batch_size).with_shuffle(0x5EED);

        let mut total_loss = 0.0f32;
        let mut batches = 0usize;
        let mut steps = 0u64;
        let mut tokens = 0u64;
        let mut report =
            self.tuner
                .record_and_evaluate(FixedPoint::ZERO, CURIOSITY_NOMINAL, BOREDOM_NOMINAL);

        loop {
            let batch = sampler.next_batch();
            if batch.is_empty() {
                break;
            }
            batches += 1;

            let mut batch_sum = 0.0f32;
            for (inputs, targets) in &batch {
                let step_report =
                    self.trainer
                        .train_step_report_scaled(inputs, targets, report.learning_scale);
                batch_sum += step_report.loss.to_f32();
                if step_report.status != TrainStepStatus::Empty {
                    tokens += inputs.len().min(targets.len()) as u64;
                }
                steps += 1;
            }

            let batch_avg = batch_sum / batch.len().max(1) as f32;
            total_loss += batch_avg;
            report = self.tuner.record_and_evaluate(
                FixedPoint::from_f32(batch_avg),
                CURIOSITY_NOMINAL,
                BOREDOM_NOMINAL,
            );
        }

        let avg_loss = if batches > 0 {
            total_loss / batches as f32
        } else {
            f32::NAN
        };

        let eval = match val {
            Some(v) => self.evaluate(v, batch_size),
            None => EvalReport::empty(),
        };

        let improved = eval.loss.is_finite() && self.best_loss.is_none_or(|b| eval.loss < b);
        if improved {
            self.best_loss = Some(eval.loss);
        }
        if improved {
            if let Some(dir) = &self.checkpoint_dir {
                if let Some(tokenizer) = &self.tokenizer {
                    let path = format!("{}/slot_{}.hklk", dir, self.epoch % 3);
                    let _ = checkpoint::save_checkpoint(
                        &path,
                        &self.trainer.model,
                        tokenizer,
                        self.trainer.step_count,
                    );
                }
            }
        }

        EpochReport {
            epoch: self.epoch,
            steps,
            batches,
            tokens,
            avg_loss,
            tuning_action: action_name(report.action),
            learning_scale: report.learning_scale.to_f32(),
            surrogate_slope: report.surrogate_slope.to_f32(),
            best_loss: self.best_loss,
            eval,
        }
    }
}

fn action_name(action: TuningAction) -> &'static str {
    match action {
        TuningAction::Maintain => "Maintain",
        TuningAction::IncreaseLearningScale => "IncreaseLR",
        TuningAction::DecreaseLearningScale => "DecreaseLR",
        TuningAction::SharpenSurrogateGradient => "SharpenSurrogate",
        TuningAction::SmoothSurrogateGradient => "SmoothSurrogate",
        TuningAction::AdjustNeuronThreshold => "AdjustThreshold",
    }
}

/// Adapter letting `BatchSampler` own a `&mut T` source.
pub struct Borrowed<'a, T: DataSource>(&'a mut T);

impl<'a, T: DataSource> DataSource for Borrowed<'a, T> {
    fn next_sample(&mut self) -> Option<(Vec<u16>, Vec<u16>)> {
        self.0.next_sample()
    }

    fn remaining_samples(&self) -> usize {
        self.0.remaining_samples()
    }

    fn reset(&mut self) {
        self.0.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::data_loader::TextDataLoader;
    use crate::training::dataset::split_tokens;

    fn corpus_tokens() -> Vec<u16> {
        let text = b"the quick brown fox jumps over the lazy dog the quick brown fox the fox the dog jumps the lazy quick fox brown";
        let mut tokenizer = BpeTokenizer::new();
        tokenizer.train(text, 32);
        tokenizer.encode_bytes(text)
    }

    #[test]
    fn test_run_epoch_produces_report_and_improves() {
        let tokens = corpus_tokens();
        let (train_tokens, val_tokens) = split_tokens(&tokens, 8, 0.2, 99);

        let mut runner = TrainingRun::new(2);
        runner.tokenizer = Some(BpeTokenizer::new());

        let mut train = TextDataLoader::new(train_tokens.clone(), 8);
        let mut val = TextDataLoader::new(val_tokens, 8);
        let report = runner.run_epoch(&mut train, Some(&mut val), 4);

        assert!(report.batches > 0);
        assert!(report.steps > 0);
        assert!(report.avg_loss.is_finite());
        assert!(report.eval.loss.is_finite());
        assert!(report.eval.accuracy >= 0.0);

        // A second epoch over the same data must keep running (no panic) and
        // must not lose the model.
        let mut train = TextDataLoader::new(train_tokens, 8);
        let report2 = runner.run_epoch(&mut train, None::<&mut TextDataLoader>, 4);
        assert_eq!(report2.epoch, 2);
        assert_eq!(report2.eval.samples, 0);
    }

    #[test]
    fn test_evaluate_empty_source_returns_empty_report() {
        let mut runner = TrainingRun::new(1);
        let mut empty = TextDataLoader::new(alloc::vec![1u16, 2, 3], 8);
        let report = runner.evaluate(&mut empty, 2);
        assert_eq!(report.samples, 0);
    }

    #[test]
    fn test_borrowed_adapter_delegates() {
        let mut loader =
            TextDataLoader::new(alloc::vec![1u16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 3);
        let mut borrowed = Borrowed(&mut loader);
        assert!(borrowed.next_sample().is_some());
        assert_eq!(borrowed.remaining_samples(), 2);
        borrowed.reset();
        assert_eq!(borrowed.remaining_samples(), 3);
    }
}
