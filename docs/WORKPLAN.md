# HKL-1 Work Plan

This plan turns HKL-1 from an ambitious embedded neuromorphic prototype into a dependable bare-metal engine. The order is risk-first: build integrity, no_std correctness, runtime safety, then performance and release polish.

## Current Baseline

- `cargo test --features std` passed before this work with the current test suite.
- The crate covers fixed-point math, static memory, SNN execution, cognitive modules, bio modules, I/O, safety, telemetry, swarm, audio, vision, NLP, system, and eFPGA generation.
- The worktree has no committed baseline yet, so changes should stay small, explicit, and verified locally.
- The main mismatch found during audit was that some runtime APIs used `alloc` even though the project presents a bare-metal/no_std profile.

## Phase 1 - Build Integrity

- Keep `cargo test --features std` green.
- Keep `cargo check --no-default-features` green for allocator-free bare-metal validation.
- Add CI jobs for default, `std`, and no-default feature sets.
- Keep Clippy focused on correctness, suspicious code, and performance as hard gates, with broad style cleanup as warnings.
- Add `cargo fmt --check` to the normal verification path.

## Phase 2 - no_std Hardening

- Remove accidental heap allocation from runtime-visible APIs.
- Gate host-only convenience APIs behind `alloc` or `std`.
- Use fixed-capacity formatting for telemetry, XAI, and HDL generation.
- Review every `unsafe` singleton and document its initialization and concurrency contract.
- Add target checks for ARM Cortex-M and RISC-V when the toolchains are installed.

## Phase 3 - Runtime Correctness

- Add reset helpers for global state used by tests and demos.
- Replace panic-prone indexing with bounded access in ISR, telemetry, persistence, and hardware paths.
- Add invariants for adaptive memory, neuron/synapse capacity, and spike queue overflow.
- Strengthen circular buffer tests for wrapped non-contiguous data.
- Expand deterministic tests for fixed-point saturation, conversion, and transcendental approximations.

## Phase 4 - Safety And Persistence

- Define explicit safety-state transitions for entropy monitor, watchdog, reflex override, OTA, and rollback.
- Add persistence compatibility tests for versioned dump headers and corrupted slots.
- Verify plaintext and encrypted persistence independently.
- Add fault injection for ECC, bad sectors, failed OTA writes, and rollback recovery.
- Define the emergency behavior when safety modules disagree.

## Phase 5 - Performance

- Benchmark `Network::step`, `step_parallel`, plasticity decay, telemetry export, and eFPGA generation.
- Add regression budgets for host simulation and embedded profiles.
- Measure memory footprint per feature set.
- Keep host threading isolated from bare-metal execution paths.
- Optimize hot loops only after benchmark evidence.

## Phase 6 - API And Architecture

- Stabilize public module boundaries and mark experimental APIs.
- Add a compact embedded prelude for common use.
- Split host simulation helpers from firmware runtime code.
- Create examples for each supported target profile.
- Version telemetry and XAI export formats.

## Phase 7 - Documentation And Release Readiness

- Fix README encoding and align claims with verified commands.
- Replace absolute marketing claims with measured constraints.
- Add architecture diagrams for data flow, safety flow, and persistence recovery.
- Add a release checklist covering targets, tests, docs, security, examples, and changelog.
- Keep CHANGELOG entries tied to verified changes.

## Phase 8 - Future Intelligence Track

- Keep HKL-2 transformer, embedding, and trainer work behind explicit feature gates until build, safety, and memory budgets are stable.
- Define an embodied cognition loop that joins SNN state, symbolic graph, temporal memory, curiosity, and safety arbitration before adding larger generative behavior.
- Add deterministic evaluation tasks for planning, tool-use simulation, continual learning, and anomaly recovery.
- Require every agent-like behavior to expose telemetry, causal traces, rollback points, and bounded resource use.
- Treat "AGI-like" milestones as measured capability ladders, not marketing claims: perception, memory, planning, self-monitoring, safe action, then open-ended synthesis.
- Keep transformer sequence lengths clamped at module boundaries and report truncation through training telemetry.
- Ship small executable cognition loops before larger demos: tokenizer, data loader, bounded forward pass, training report, then safety review.
- Add metacognitive guards that convert model telemetry into continue, throttle, or halt decisions before adaptive updates scale up.
- Add an executive cognition loop that arbitrates learning, exploration, consolidation, recovery, and idle behavior from telemetry plus safety pressure.
- Convert executive decisions into bounded cognitive plans with explicit step budgets, checkpoint requirements, rollback paths, and trace identifiers.
- Compose guard, executive, and planner into a reusable cognition controller so examples and future runtimes share one authorization path.
- Emit compact cycle audit records with risk level, flags, trace id, plan summary, and learning authorization for telemetry handoff.
- Add deterministic cognition scenarios that exercise learning, exploration, budget starvation, saturated-loss recovery, and safety-pressure recovery.
- Maintain deterministic HKL-2 scenario evaluations that verify action, risk, recovery, checkpoint, learning budget, and external-effect authorization before larger loops are trusted.
- Report truncated HKL-2 scenario suites explicitly so skipped safety cases cannot masquerade as a fully passing evaluation.
- Include requested scenario counts and proportional skipped-case penalties in HKL-2 eval reports so oversized suites remain visibly incomplete.
- Normalize HKL-2 readiness policies before use so external threshold configuration cannot invert maturity gates or demand impossible recovery evidence.
- Make HKL-2 agentic-loop permission recompute report consistency from counters and evidence, not just trust level/flag fields.
- Score HKL-2 scenario suites with per-mille capability plus learning, exploration, recovery, and restraint sub-points.
- Gate HKL-2 larger agent-like loops behind explicit readiness levels: blocked, observe-only, learning-ready, adaptive-ready, and agentic-candidate.
- Combine global HKL-2 readiness with live cycle audits through an operational runtime gate so mature scenario scores cannot override current recovery, budget, or checkpoint facts.
- Track HKL-2 runtime gate decisions across bounded episodes with a no-alloc supervision ledger before longer autonomous loops are allowed to run unattended.
- Route HKL-2 unattended loops through a reusable episode runner that composes controller, audit, runtime gate, supervision, and final recommendation before any model update.
- Make HKL-2 runtime-gate permission helpers revalidate flags, live risk, learning budget, and learning scale so forged public decisions cannot enable effects.
- Make HKL-2 supervision ledgers count effective runtime permissions rather than trusting raw public gate modes.

## Immediate Work Started

- Added fixed-capacity text formatting support for no_std exports.
- Removed allocator-dependent formatting from spike trace, XAI, and eFPGA HDL generation.
- Gated dynamic XAI path reconstruction behind `alloc` or `std`.
- Replaced hippocampus WTA sorting with a fixed-size top-k selection suitable for no-default builds.
- Added CI build-integrity checks for default, `std`, allocator-free no-default, and `hkl2` feature profiles.
- Aligned host Clippy/tests with the `hkl2` learning feature so experimental learning code stays covered.
- Added a CI run for the minimal prelude example to keep the documented public API smoke-tested.
- Hardened spike trace wraparound handling so telemetry exports no longer build an invalid cross-boundary slice.
- Added chronological trace iteration/copy helpers and routed telemetry/XAI analysis through the full logical ring-buffer order.
- Added wraparound regression tests for spike trace iteration, UART event counts, and full-copy ordering.
- Hardened HKL-2 training/tokenization inputs: zero-length data-loader sequences now terminate safely, sample counts are explicit, and BPE rejects ambiguous or recursive merge rules.
- Verified the expanded `std,hkl2` profile with targeted embedding/training tests and a full single-threaded test pass.
- Exported the BPE tokenizer module and made merged-token decoding reversible for known merge rules.
- Hardened spiking transformer forward paths so oversized requested sequence lengths clamp to available input.
- Aligned HKL-2 trainer accounting with the transformer's bounded logits and kept zero-length data loader samples non-advancing.
- Added the HKL-2 training-loop example to CI and aligned public quick-start commands with the expanded feature profile.
- Clamped HKL-2 spiking cross-entropy to non-negative firing rates so negative logits cannot produce misleading probabilities or learning signals.
- Split HKL-2 training preview from adaptive updates so guards can halt or scale learning before weights change.
- Bounded HKL-2 executive outputs and causal trace intensities so externally supplied telemetry cannot produce unbounded learning or exploration commands.
- Hardened HKL-2 training guard policy and ratio accounting against out-of-range thresholds and externally supplied oversized counters.
- Routed flash persistence writes away from firmware banks by using the OTA-declared persistence base address and tested the layout invariant.
- Made persistence `load_slot` verify magic, version, checksum, and count bounds before restore, with corruption/count regression tests.
- Serialized persistence tests and public persistence entrypoints under `cfg(test)` so shared dump slots stay deterministic in parallel Cargo test runs.
- Replaced ISR pending-flag `load`/`store(0)` with an atomic drain operation so deferred interrupts are not lost between polling and clear.
- Serialized ISR tests that share pending flags and the global spike queue so host test parallelism cannot leave cross-test queue residue.
- Added an explicit clear operation for the global ISR spike queue and routed buffer initialization/tests through it.
- Reworked `secure_erase` to use volatile byte writes plus compiler fences and added a slice-level erase helper.
- Added an HKL-2 training loop example wiring BPE tokenization, autoregressive samples, transformer training, and `TrainStepReport` telemetry.
- Hardened HKL-2 transformer/trainer sequence bounds so oversized inputs clamp to `MAX_SEQ_LEN` consistently.
- Hardened HKL-2 layer normalization so oversized dimensions and mismatched runtime slices clamp safely instead of panicking.
- Hardened HKL-2 attention linear transforms so short external buffers are rejected without panics or partial output mutation.
- Added trainer status telemetry for empty, complete, and truncated steps plus invalid-target/saturated-loss accounting.
- Rejected BPE merge rules that reference unknown high token ids, preventing decode-only dead ends.
- Added a dedicated Cargo test profile so verification does not inherit fat-LTO firmware build settings.
- Treated out-of-vocabulary HKL-2 input tokens as silent positions and reported invalid input/target counts without applying corrupted training updates.
- Aligned HKL-2 spike embedding with the OOV contract so direct out-of-vocabulary encodes reset state and return silent spike trains.
- Added explicit HKL-2 state reset APIs from spike embeddings through transformer blocks and trainer episodes, with regression tests for membrane cleanup.
- Isolated HKL-2 preview training reports so guard/executive dry-runs reset transient spiking state before and after evaluation.
- Normalized executive cognition policies at loop construction so external thresholds remain bounded before they can steer learning, exploration, or recovery.
- Compacted HKL-2 cognitive plans after budget clamping so zero-budget mutating steps are not advertised or executed.
- Kept HKL-2 recovery mandatory at the controller level even when tight plan budgets trim the explicit recovery step.
- Preserved HKL-2 rollback/checkpoint requirements across tight cognitive plan budgets so recovery intent remains auditable.
- Added a HKL-2 `TrainingGuard` that evaluates `TrainStepReport` telemetry and emits bounded continue/throttle/halt decisions for future cognition loops.
- Added a HKL-2 `ExecutiveLoop` that turns trainer reports, guard decisions, curiosity, novelty, prediction error, and safety pressure into auditable bounded actions.
- Added a HKL-2 `CognitivePlanner` that converts executive actions into fixed-size plans for safety checks, probes, learning, consolidation, recovery, and telemetry.
- Added a HKL-2 `CognitiveController` that composes dry-run trainer reports, guard decisions, executive arbitration, planning, and final learning authorization.
- Added a HKL-2 `CycleAuditRecord` that summarizes each cognition cycle into stable risk/flag/trace fields for future telemetry and persistence.
- Added a HKL-2 scenario suite that evaluates controller/audit behavior across five deterministic agent-loop cases and reports pass/fail plus a summary hash.
- Hardened HKL-2 cycle audit records so learning authorization and plan length summaries are recomputed from bounded effective cycle facts.
- Strengthened HKL-2 scenario evaluation so regressions in external-effect authorization, checkpoint requirements, and learning-budget availability are reported as first-class mismatches.
- Expanded HKL-2 scenario summary hashing to include bounded audit facts such as pass status, action, risk, first step, budgets, plan length, and tokens seen.
- Hardened HKL-2 scenario suite reporting so inputs beyond the fixed scenario capacity are counted as skipped and prevent an all-passed interpretation.
- Made HKL-2 scenario scores penalize every skipped scenario, with the requested/evaluated counts surfaced in the executable training-loop summary.
- Hardened HKL-2 readiness evaluation by clamping policy thresholds to 0..1000, preserving learning/adaptive/agentic ordering, and capping required recovery evidence to scenario capacity.
- Hardened `ReadinessReport::permits_agentic_loop` so public or forged reports must still prove evaluated=requested, passed=evaluated, zero failures/skips, perfect score, and non-zero learning/exploration/recovery/restraint evidence.
- Added HKL-2 scenario scoring so deterministic evals report capability per mille and separate learning/exploration/recovery/restraint points.
- Added HKL-2 readiness gates that convert deterministic scenario scores into an explicit maturity level and blocking reason flags.
- Added an HKL-2 runtime gate that combines readiness reports with live cycle audits before learning or exploration can proceed.
- Added an HKL-2 supervision ledger that summarizes multi-cycle learning, exploration, recovery, blocking, risk, streak, and trace-hash evidence.
- Added an HKL-2 episode runner that emits explicit apply-learning, probe, recover, observe, or stop recommendations from the shared cognition authorization path.
- Hardened `RuntimeGateDecision` permission helpers so learning/exploration/recovery decisions require internally consistent flags, risk, budget, and scale evidence even if the public decision struct is constructed manually.
- Hardened `SupervisionLedger::record` so forged runtime decisions without effective permission or readiness-block evidence are observed instead of counted as learning, recovery, or blocked.
- Hardened HKL-2 runtime gate diagnostics so learning requests without external-effect authorization are reported instead of silently falling back to observation.
