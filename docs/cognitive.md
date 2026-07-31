# Cognitive Module

The cognitive module implements higher-level functions for goal-directed behavior and learning.

## Actor-Critic (`cognitive/actor.rs`)

Reinforcement learning with separate policy (actor) and value (critic) networks.

- **Actor**: Policy gradient for action selection with ε-greedy exploration (decay 0.3 → 0.01)
- **Critic**: State-value estimation via TD learning (64 buckets V(s))
- **TD-error**: `compute_td_error()` — reward prediction error as dopamine signal
- **Reward shaping**: `compute_reward()` based on prediction error + novelty + energy
- **Hypotheses**: `generate_hypotheses()` generates candidate actions for predictive cycle

## Attention (`cognitive/attention.rs`)

Saliency-based attention mechanism for sensory filtering.

- Bottom-up saliency (stimulus-driven) via SaliencyMap per layer
- Top-down attention (goal-driven) with action→layer routing, gain/suppression, focus dwell
- Winner-take-all selection per layer (max → inhibit neighbors)

## Curiosity (`cognitive/curiosity.rs`)

Intrinsic motivation engine for exploration.

- Prediction-error-based novelty
- Adaptive exploration rate (`explore_epsilon()` based on boredom + monotony + curiosity)
- Dreaming mode for offline learning (`activate_dreaming()`)
- Habituation to familiar stimuli (HabituationTracker 32 slots, sigmoid familiarity)
- Thermal noise injection → sensory layer (proportional to temperature)
- Boredom accumulator + monotony counter for anti-stagnation

## Predictor (`cognitive/predictor.rs`)

Forward world model for prediction and simulation using prototype-based learning.

- Prototypes by action: each known action has a `Prototype` with `delta_predictions` and `confidence`
- `predict(&state_array)` → uses `predict_next(&state_array, self.last_action)` to compute delta
- `last_action` field tracks the most recent action for state transition prediction
- `learn_prototype(action, obs, reward)` — online Hebbian learning with EMA confidence update
- `update_from_prediction_error(action, error)` — merges prototypes, decays confidence for stale entries
- Transition buffer (256 entries) for experience replay
- Connected to `predictive_cycle()` in Network for mental simulation via TimeWarper

### Prototype Merging

When two prototypes for the same action have similar deltas, they are merged:
- Delta vectors averaged
- Confidence combined as `max(c1, c2)`
- Reduces memory usage while retaining accuracy

## Neuromodulation (`cognitive/neuromodulation.rs`)

Central neuromodulatory system:

| Modulator | Source | Function |
|---|---|---|
| Dopamine | VTA-like | Reward prediction error, learning gate |
| Serotonin | Raphe-like | Mood regulation, risk assessment |
| Noradrenaline | LC-like | Arousal, novelty detection |
| Acetylcholine | Basal forebrain | Attention, memory encoding |

### Calibration

`NeuromodulationCalibration` auto-tunes parameters:
- EMA prediction error tracking
- Volatility estimation
- Adaptive decay rates
- Auto-adjusted LTP/LTD sensitivity

### Neuromodulator Synchronization

Flows: **TD-error → SNN dopamine → Cognitive dopamine** (bidirectional)
- `COGNITIVE_NEUROMODULATORS.sync_to_snn()` copies cognitive state to `GLOBAL_NEUROMODULATORS`
- `GLOBAL_NEUROMODULATORS` provides dopamine concentration to SNN plasticity
- Inverse flow: SNN spike activity modulates neuromodulator release

## Proprioception (`cognitive/proprioception.rs`)

Body-state tracking through efference copy and prediction error.

- Forward model of motor commands via `record_efference()`
- Sensory reafférence prediction
- Error-driven corrective signals: `apply_correction()` injects current into layer 4 + bias, triggers NA/ACh
- Body model learning: 64 `BodyModelEntry` slots with online Hebbian weight adjustment and accuracy tracking

## Temporal Cognition (`cognitive/temporal.rs`)

Time-cell representations for sequence learning and interval timing.

- **Time cells**: 64 offsets from 1ms to 50s with Gaussian activation profiles and decay
- **Sequence buffer**: Circular buffer (256 entries) for pattern detection (3-action patterns)
- **Interval timing**: `IntervalTimer` with `target_ms`, `elapsed`, `fired` flag, `reset()`
- **Pattern prediction**: `predict_next_action()` uses `pattern.last_triggered_ms + pattern.intervals[0]` for real timing (not synthetic offset)
- **Multi-scale integration**: 5 temporal buffers (ultrafast → ultraslow) read via `MetabolicClock`
- `TemporalPattern.last_triggered_ms` tracks real-world timing for interval-based prediction

## Networks Calibration (`cognitive/networks.rs`)

`NeuromodulationCalibration` auto-tunes dopamine, noradrenaline, serotonin, and acetylcholine target levels based on running statistics (EMA of prediction error, reward, novelty, volatility).

- `update(pred_error, reward, novelty)` — online EMA update; recomputes sensitivity LTP/LTD, decay rate, and volatility estimate
- `calibrate_neuromodulators(now)` — runs every 50 ticks; reads calibration state, computes target levels for each modulator, blends toward global state (90% old / 10% new)
- `adaptive_decay_rate()` — returns current decay rate clamped to `[0.0005, 0.01]`

### Calibration Flow

1. Prediction error, reward, and novelty are fed to `NeuromodulationCalibration::update()`
2. Running mean and variance update via EMA ($\alpha = 0.05$)
3. Volatility estimate: ratio of prediction error variance to total variance
4. Sensitivity LTP/LTD: sigmoid mapping of reward mean to `[0.2, 0.9]`
5. `calibrate_neuromodulators()` reads these metrics and computes target modulator levels (DA from reward, NA from novelty, 5-HT from volatility, ACh from prediction error)

## Reflex Override (`cognitive/reflex_override.rs`)

Cognitive override of hardwired spinal reflexes when the system is in a stable, focused state.

- `evaluate_override() -> bool`: Returns `true` only when all three conditions hold:
  1. Noradrenaline ≤ 0.6 — no crisis/survival mode
  2. Cognitive mode is `Stable` — not in exploration or chaos
  3. Attention dwell counter ≥ 5 — focus has stabilized on a target
- `override_attenuation() -> FixedPoint`: Returns `1.0` (no suppression) if override not granted, else `1.0 - NA` — calmer states suppress reflexes more strongly

## Episodic Memory (`cognitive/episodic.rs`)

Dual-store hippocampus-like memory system with consolidation, replay, and spatial navigation.

### Memory Architecture
- **Short-term buffer** (256 traces): Fast-decay, high plasticity for recent experiences
- **Long-term store** (512 traces): Slow-decay, consolidated via significance-based transfer
- **Ebbinghaus forgetting curve**: Retention $R(t) = 2^{-t/\tau}$ with half-life $\tau = 10^4$ (ST) / $10^6$ (LT)

### Consolidation & Replay
- **Significance scoring**: $S = 0.4\cdot\text{reward} + 0.35\cdot\text{PE} + 0.25\cdot\text{novelty}$
- **Offline consolidation**: Top-scoring ST traces transferred to LT each metabolic cycle
- **Prioritized replay**: `sample_replay_batch()` returns experiences weighted by significance
- **Sharp-wave ripple replay**: `trigger_ripple_replay()` replays significant memories in reverse order during rest/sleep, boosting access times

### Spatial Navigation
- **Place cells** (128): Gaussian place fields distributed across 8×8 grid, activated by spatial position
- **Grid cells** (64): Entorhinal-style hexagonal grid with 5 scales, hexadirectional firing
- **Theta phase precession**: Firing phase shifts proportionally to distance from place field center
- **Path integration**: Velocity-driven position update with clamp to unit square

### Bridge to Bio Hippocampus
The episodic memory is bidirectionally linked to `bio/hippocampus`:

1. **Bio→Cognitive**: When the bio hippocampus detects SWR events (CA3 active > 25% + correct theta phase), it triggers `trigger_ripple_replay()` in episodic memory, consolidating spatial context into long-term storage
2. **Cognitive→Bio**: Place cell activity from episodic memory enriches the sensory input fed to the bio hippocampus (30% blend, `hipp_input[i] += place_rates[i] * 0.3`)

## Continual Learning & Anti-Catastrophic Forgetting (`cognitive/continual.rs`)

Integrated continual learning engine preventing catastrophic forgetting while enabling rapid adaptation:

- **Offline Replay (`OfflineReplayEngine`)**: Sharp Wave-Ripples (SWR 150-250 Hz) hippocampal experience replay during rest/sleep phases to consolidate short-term memories into long-term storage. Triggered by bio hippocampus SWR events.
- **Few-Shot Learning (`FewShotAdapter`)**: Fast-Weights 4x STDP plasticity booster for rapid 1-to-3 shot adaptation when novel task patterns are encountered.
- **Meta-Learning Auto-Tuning (`MetaLearningEngine`)**: Dynamic performance-driven tuning of learning rate $\eta_{\text{STDP}}$, dopamine threshold $\theta_{\text{DA}}$, and decay constants based on global prediction error.
- **Elastic Weight Consolidation (`ElasticWeightConsolidation`)**: Fisher Information Matrix $F_{ij}$ computation and EWC penalty opposing perturbation of critical task synapses, achieving 0% catastrophic forgetting.

## Biological Brain Macro-Structures (`src/bio/`)

Detailed biological sub-modules synchronized with the SNN step loop:

- **Thalamus (`bio/thalamus.rs`)**: Sensory gating relay (4 channels) modulating sensory signal transmission into Layer 0 based on top-down attention and prediction errors.
- **Striosome Matrix (`bio/striosome.rs`)**: Basal ganglia action selection engine (16 striosomes, 64 matrix units) applying dopamine-gated reinforcement learning.
- **Hippocampus (`bio/hippocampus.rs`)**: 256 Dentate Gyrus (DG) pattern separators, 64 CA3 recurrent auto-associative units, and 64 CA1 output neurons with Sharp Wave-Ripple (SWR) detection.
- **Astrocyte Network (`bio/astrocytes.rs`)**: 64 glial astrocyte cells propagating slow intercellular calcium waves ($\sim 0.1 \text{ mm/sec}$) to dynamically modulate local synaptic thresholds.
- **Cerebellum (`bio/cerebellum.rs`)**: 64 Purkinje cells and inferior olive climbing fiber error-driven motor command refinement.

