use hkl1::cognition::controller::CycleSignals;
use hkl1::cognition::episode::CognitiveEpisodeRunner;
use hkl1::cognition::readiness::evaluate_readiness;
use hkl1::cognition::scenario::run_default_scenarios;
use hkl1::embedding::bpe_tokenizer::BpeTokenizer;
use hkl1::training::data_loader::TextDataLoader;
use hkl1::training::trainer::Trainer;

fn main() {
    let scenario_report = run_default_scenarios();
    let readiness = evaluate_readiness(&scenario_report);
    println!(
        "HKL-2 scenarios requested={} evaluated={} passed={} failed={} skipped={} score={}/1000 readiness={:?} ready={} flags=0x{:04x} learn={} explore={} recover={} restraint={} hash={:08x}",
        scenario_report.requested,
        scenario_report.len,
        scenario_report.passed,
        scenario_report.failed,
        scenario_report.skipped,
        scenario_report.score.capability_per_mille,
        readiness.level,
        readiness.permits_agentic_loop(),
        readiness.flags.0,
        scenario_report.score.learning_points,
        scenario_report.score.exploration_points,
        scenario_report.score.recovery_points,
        scenario_report.score.restraint_points,
        scenario_report.summary_hash
    );

    let mut tokenizer = BpeTokenizer::new();
    tokenizer.add_merge(b'H' as u16, b'K' as u16, 256);
    tokenizer.add_merge(256, b'L' as u16, 257);

    let tokens = tokenizer.encode_bytes(b"HKL future intelligence loop");
    let mut loader = TextDataLoader::new(tokens, 4);
    let mut trainer = Trainer::new(0);
    let mut episode_runner = CognitiveEpisodeRunner::conservative(readiness, 16);
    let mut cycle_index = 0u32;

    while cycle_index < 3 {
        let Some((inputs, targets)) = loader.next_sample() else {
            break;
        };
        let preview = trainer.preview_step_report(&inputs, &targets);
        let signals = CycleSignals::neutral();
        let episode = episode_runner.preview_cycle(preview, signals);
        let report = if episode.allows_model_update() {
            trainer.reset_model_state();
            trainer.train_step_report_scaled(&inputs, &targets, episode.runtime_gate.learning_scale)
        } else {
            episode.cycle.preview_report
        };
        println!(
            "HKL-2 cycle={} step={} tokens={} invalid={}/{} saturated={} status={:?} guard={:?}/{:?} lr_scale={:.2} exec={:?}/{:?} risk={:?} audit_flags=0x{:04x} runtime={:?} runtime_flags=0x{:04x} supervision={:?} recommendation={:?} plan_len={} first={:?} budget={} learn_budget={} rollback={} checkpoint={} trace={:08x} loss={:.3}",
            cycle_index,
            trainer.step_count,
            report.tokens_seen,
            report.invalid_inputs,
            report.invalid_targets,
            report.saturated_losses,
            report.status,
            episode.cycle.guard_decision.action,
            episode.cycle.guard_decision.reason,
            episode.runtime_gate.learning_scale.to_f32(),
            episode.cycle.executive_decision.action,
            episode.cycle.executive_decision.reason,
            episode.audit.risk,
            episode.audit.flags.0,
            episode.runtime_gate.mode,
            episode.runtime_gate.flags.0,
            episode.supervision.status,
            episode.recommendation,
            episode.cycle.plan.len,
            episode.cycle.plan.first_step().kind,
            episode.cycle.plan.total_budget_ticks,
            episode.audit.learning_budget_ticks,
            episode.cycle.must_recover,
            episode.cycle.checkpoint_required,
            episode.cycle.plan.trace_id,
            report.loss.to_f32()
        );
        cycle_index += 1;
        if episode.should_stop_unattended() {
            break;
        }
    }

    let supervision = episode_runner.supervision();
    if supervision.cycles == 0 {
        println!("HKL-2 no training sample available");
    } else {
        println!(
            "HKL-2 supervision cycles={} status={:?} learn={} explore={} observe={} blocked={} recover={} streak={} max_streak={} max_risk={:?} last={:?} flags=0x{:04x} hash={:08x}",
            supervision.cycles,
            supervision.status,
            supervision.learning_allowed,
            supervision.exploration_allowed,
            supervision.observed,
            supervision.blocked,
            supervision.recovery,
            supervision.recovery_streak,
            supervision.max_recovery_streak_seen,
            supervision.max_risk,
            supervision.last_mode,
            supervision.flags.0,
            supervision.summary_hash
        );
    }
}
