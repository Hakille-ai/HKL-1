# Roadmap HKL-1 — Analyse complète vs Document de Conception Technique (TDD)

> **Légende :** ✅ Implémenté — ⚠️ Partiel / à enrichir — ❌ Pas implémenté — 📅 Planifié
>
> Références : **Section N** = section du Document de Conception Technique (TDD)

---

## Phase 1 — Fondations neuromorphiques (Noyau SNN)

Ces modules constituent le socle obligatoire pour tout le reste.

### 1.1 CORE — Mathématiques & Mémoire

| Composant | TDD | Statut | Fichier | Écart |
|---|---|---|---|---|
| FixedPoint Q16.16 | — | ✅ | `core/math.rs` | Complet (exp, ln, sqrt, pow) |
| Weight Q8.8 | — | ✅ | `core/math.rs` | Complet |
| XorShift64Star PRNG | — | ✅ | `core/math.rs` | Complet |
| Matrix/Vector | — | ✅ | `core/math.rs` | Matrices carrées taille fixe |
| NeuronState / SynapseSlot | — | ✅ | `core/memory.rs` | Complet |
| StaticPool / GlobalPool | — | ✅ | `core/memory.rs` | Allocateur bitfield + liste libre |
| NEURON_ARRAY / SYNAPSE_ARRAY | — | ✅ | `core/memory.rs` | MaybeUninit, zéro-coût |

### 1.2 SNN — Neurones & Synapses

| Composant | TDD | Statut | Fichier | Écart |
|---|---|---|---|---|
| LIFNeuron (τ dV/dt = -V + I) | §3.1 | ✅ | `snn/neuron.rs` | Intégration, fuite, spike, réfractaire |
| Types de neurones (6) | — | ✅ | `snn/neuron.rs` | Excitatory, Inhibitory, Modulatory, Pacemaker, Sensory, Motor |
| Synapse avec R-STDP | §5.3 | ✅ | `snn/synapse.rs` | Poids, délai, dépression, facilitation |
| STDP (Δw = A₊ exp(-Δt/τ₊)...) | §5.3 | ✅ | `snn/plasticity.rs` | Traces pré/post, fenêtre asymétrique |
| Plasticité modulée par dopamine (M(t)) | §5.3 | ✅ | `snn/plasticity.rs` | Calibration complète DA/NA/5-HT/ACh via `calibrate_neuromodulators()` |
| Traces d'éligibilité | — | ✅ | `snn/plasticity.rs` | Pour crédit temporel |

### 1.3 SNN — Réseau & Homéostasie

| Composant | TDD | Statut | Fichier | Écart |
|---|---|---|---|---|
| Network::step() / step_parallel() | — | ✅ | `snn/network.rs` | Propagation, mise à jour multi-threadée parallèle (`std::thread::scope`) |
| ActorNetwork / PredictorNetwork | §6 | ✅ | `snn/network.rs` | Prédicteur utilise prototypes par action + confidence scaling, predictive_cycle() connectée |
| Homéostasie (taux de décharge cible) | §20 | ✅ | `snn/homeostasis.rs` | Scaling synaptique, compensation par couche |
| Neurogenèse (élagage + création) | §12 | ✅ | `snn/neurogenesis.rs` | Pool libre, recyclage, adjacence |
| Seuil de pruning (w < ε, N cycles) | §12 | ✅ | `snn/neurogenesis.rs` | Paramètres configurables |
| HardwareDetector | — | ✅ | `system/hardware_detect.rs` | Détection automatique des cœurs CPU et RAM système |
| ADAPTIVE_MEMORY | — | ✅ | `core/memory.rs` | Mise à l'échelle adaptative dynamique des capacités SNN |


### 1.4 Temps & Métabolisme

| Composant | TDD | Statut | Fichier | Écart |
|---|---|---|---|---|
| MetabolicClock (1Hz–1MHz) | §18 | ✅ | `core/time.rs` | 5 fréquences, heartbeat 1Hz |
| TemporalHierarchy (5 buffers) | §33 | ✅ | `core/time.rs` | Ultra-rapide → Ultra-lent, 1024 entrées |
| TimeWarper (accélération ×1000) | §6 | ✅ | `core/time.rs` | Pour simulation prédictive |

---

## Phase 2 — Persistance, Sécurité & Boot

### 2.1 Persistance

| Composant | TDD | Statut | Fichier | Écart |
|---|---|---|---|---|
| BinaryDump (copie bit-à-bit) | §7.1 | ✅ | `system/persistence.rs` | Header + neurones + synapses |
| 3 slots de sauvegarde | §7.2 | ✅ | `system/persistence.rs` | J-0, J-1, J-2 |
| Checksum | §7.1 | ✅ | `system/persistence.rs` | CRC32 |
| Restauration auto au boot | §9 | ✅ | `system/boot.rs` | Vérifie et restaure |

### 2.2 Sécurité & Crypto

| Composant | TDD | Statut | Fichier | Écart |
|---|---|---|---|---|
| ChaCha20 (chiffrement flux) | §10 | ✅ | `core/crypto.rs` | Implémentation from scratch |
| PUF (SRAM / Ring Oscillator) | §10 | ✅ | `core/crypto.rs` | Interface, registres mmap |
| EphemeralKeyManager | §10 | ✅ | `core/crypto.rs` | Clé jamais stockée |
| Secure Erase | §10 | ✅ | `core/crypto.rs` | XOR + vérification |
| HMAC-SHA256 | — | ✅ | `core/crypto.rs` | Pour validation firmware |

### 2.3 Boot & Watchdog

| Composant | TDD | Statut | Fichier | Écart |
|---|---|---|---|---|
| Boot sequence (t=0→22ms) | §9 | ✅ | `system/boot.rs` | Init clock, restore, main loop |
| Watchdog (actions graduées) | §7.2 | ✅ | `system/watchdog.rs` | Reset → Rollback → Restore full |
| **Custom Linker Script (.ld)** | §27 | ✅ | `stm32f746.ld`, `hifive1.ld`, `esp32c6.ld` | TCM pour matrices critiques |

---

## Phase 3 — COGNITIF — Implémenté mais insuffisant ⚠️

Ces modules existent mais sont trop petits ou incomplets par rapport au TDD.

### 3.1 Neuromodulation (§21)

| Composant | Statut | Taille | Écart |
|---|---|---|---|
| 4 hormones (DA, 5-HT, NA, ACh) | ✅ | 2.4KB | Stock + modes + decay |
| Boucle de rétroaction avec apprentissage | ✅ | networks.rs | Modulation LTP/LTD via modulate_plasticity() + apply_plasticity_modulation(), dopamine connectée au TD-error |
| Calibration automatique des taux | ✅ | networks.rs | NeuromodulationCalibration: EMA prédiction error, volatilité, decay rate adaptatif, sensitivity LTP/LTD auto-ajustée |

### 3.2 Attention / WTA (§14)

| Composant | Statut | Taille | Écart |
|---|---|---|---|
| WTA par couche (max → inhibe voisins) | ✅ | 1.9KB | ✅ Fonctionnel |
| Attention bottom-up (saillance) | ✅ | attention.rs | SaliencyMap par couche, peak neuron, bottom-up computation |
| Attention top-down (goal-driven) | ✅ | attention.rs | Action→layer routing, gain/suppression, focus dwell |

### 3.3 Curiosité & Bruit Stochastique (§29)

| Composant | Statut | Taille | Écart |
|---|---|---|---|
| Injection de bruit stochastique | ✅ | 3.7KB | PRNG/TRNG/thermique |
| Moteur de curiosité (dreaming) | ✅ | — | `activate_dreaming()` |
| **Bruit thermique CPU → spikes** | ✅ | curiosity.rs | Temp sensor → layer 0/1 noise, amplitude proportionnelle à la température |
| **Habituation / ennui artificiel** | ✅ | curiosity.rs | HabituationTracker 32 slots, familiarité sigmoïde, boredom_accumulator, monotony_counter |
| **Seuils adaptatifs d'exploration** | ✅ | curiosity.rs | explore_epsilon() adaptatif (boredom + monotony + curiosity), ε-greedy synced avec actor |

### 3.4 Proprioception (§22)

| Composant | Statut | Taille | Écart |
|---|---|---|---|
| Copie d'efférence | ✅ | 2.1KB | `record_efference()` ✅ |
| Erreur de prédiction | ✅ | — | Comparaison prédit/réel |
| **Action corrective automatique** | ✅ | proprioception.rs | `apply_correction()` injecte courant correctif dans layer 4 + bias, déclenche NA/ACh |
| **Apprentissage du modèle corporel** | ✅ | proprioception.rs | BodyModelEntry 64 slots, learned_weight ajusté par online Hebbian, accuracy tracking |

### 3.5 Entropie Cognitive (§32)

| Composant | Statut | Taille | Écart |
|---|---|---|---|
| EntropyMonitor (Shannon) | ✅ | 2.1KB | Seuils adaptatifs + CognitiveMode |
| **Seuils adaptatifs** | ✅ | core/entropy | Mean ± 2σ, smooth EMA, 4 CognitiveModes |
| **Corrélation avec neuromodulation** | ✅ | core/entropy + network | apply_cognitive_mode() → NM crise/explore/stability |

---

## Phase 4 — COGNITIF — Pas implémenté ❌

### 4.1 Actor-Critic RL (§6 — implicite) ✅ IMPLÉMENTÉ

| Composant | Taille | Statut |
|---|---|---|
| Policy network (hypothèses + sélection par valeur) | ✅ | `ActorCritic` avec `select_action()` et `generate_hypotheses()` |
| Value network (critique TD) | ✅ | Table de 64 buckets V(s) avec mise à jour TD |
| TD-error comme signal d'apprentissage | ✅ | `compute_td_error()` + `update_value_from_next()` |
| Reward shaping | ✅ | `compute_reward()` basé sur erreur prédiction + nouveauté + énergie |
| ε-greedy exploration | ✅ | Décroissance de 0.3 → 0.01 avec `Random` |
| Connexion dopamine ← TD-error | ✅ | `COGNITIVE_NEUROMODULATORS.dopamine` = clamp(δ + 1) * 0.5 |

### 4.2 Predictor Network (§6)

| Composant | Taille | Statut |
|---|---|---|
| Modèle de forward dynamics (sₜ → sₜ₊₁) | ✅ | Prototypes par action avec delta-prediction |
| Erreur de prédiction → apprentissage | ✅ | `update_from_prediction_error()` + prototype merging |
| Simulation latente (TimeWarper + Predictor) | ✅ | Connecté via `predictive_cycle()` → `run_simulation()` |
| Apprentissage en ligne (Hebb/TD) | ✅ | `learn_prototype()` avec merge + EMA confidence |

### 4.3 Temporal Cognition (§33)

| Composant | Taille | Statut |
|---|---|---|
| Cellules de temps (time cells) | ✅ | 64 offsets 1ms–50s, activation gaussienne, decay |
| Fenêtres temporelles pour séquences | ✅ | Buffer circulaire 256, détection patterns 3-actions |
| Intégration multi-échelle (fusion 5 buffers) | ✅ | Phases ultrafast→ultraslow lues via MetabolicClock |
| Interval timing | ✅ | Timer avec target, elapsed, fired, reset |

### 4.4 Cycle Prédictif Complet (§6) ✅ IMPLÉMENTÉ

Le TDD décrit un cycle : Acteur → Prédicteur → TimeWarp → Validation → Action. **Connecté via `predictive_cycle()` dans `Network::step()`.**

| Étape | Statut |
|---|---|
| 1. Acteur se déconnecte des sorties (inhibition) | ✅ `actor.output_inhibited = true` |
| 2. Acteur génère hypothèses → Prédicteur | ✅ `generate_hypotheses()` → `predict_next()` |
| 3. TimeWarper accélère simulation | ✅ `activate_warp(100)` → `run_simulation()` |
| 4. Résultat positif → reconnecte Acteur | ✅ Dopamine + `output_inhibited = false` |
| 5. Enregistrement transition | ✅ `record_transition()` pour apprentissage futur |
| 6. Cooldown après cycle | ✅ 5000 pas si succès, 2000 sinon |

---

## Phase 5 — I/O Matériel ✅

| Composant | TDD | Taille | Statut | Description |
|---|---|---|---|---|
| Sensors (I2C, SPI, ADC) | §4 | 16.0KB | ✅ | SensorManager global, `i2c_error_count: u32` tracké dans `read_sensor_i2c()` |
| Actuators (PWM, GPIO, DAC) | — | 9.2KB | ✅ | PWM/GPIO/DAC register writes via MMIO, `DacOutput::init()` écrit DAC_CR EN+BOFF |
| **Interruptions matérielles (ISR → RingBuffer)** | §23 | — | ✅ | Gestionnaires TIM2/ADC/EXTI/SPI/I2C complets, `isr_push_spike()` layer depuis intensité |
| **Gestionnaire d'interruption matérielle** | §23 | 13KB | ✅ | NVIC + handlers TIM2/ADC/EXTI/SPI/I2C → GLOBAL_SPIKE_QUEUE |
| **Connexion ISR → GLOBAL_SPIKE_QUEUE** | §23 | — | ✅ | ISR reserve_write/commit_write |
| Encoder (texte, audio, vision, capteurs) | §4 | 9.8KB | ✅ | Tous les encodeurs temporels |
| Decoder (texte, voix) | §28 | 2.7KB | ✅ | TextOutput, VoiceOutput |

---

## Phase 6 — Swarm Intelligence

| Composant | TDD | Taille | Statut | Écart |
|---|---|---|---|---|
| FederatedLearning (bruit DP) | §8 | 2.5KB | ✅ | Agrégation + bruit gaussien + topologie adaptative + fiabilité des nœuds |
| **Topologie adaptative** | §8 | federated.rs | ✅ | Topologie cache basée sur reliability, mise à jour périodique |
| **Découverte de pairs** | §8 | mesh.rs | ✅ | Heartbeat, timeout, ajout/suppression automatique |
| MeshNetwork (deltas de poids) | §8 | 5.2KB | ✅ | Calcul Δw + application + découverte + gossip + clock sync + remote spikes |
| **Gossip protocol** | §8 | mesh.rs | ✅ | File d'attente 64 messages, fanout=3, TTL 1000ms |
| **Sync d'horloge entre nœuds** | §8 | mesh.rs | ✅ | Clock offset tracking, drift moyen, RTT |
| **Messages spike distants** | §8 | mesh.rs | ✅ | RemoteSpike avec neuron_idx + amplitude, application dans le réseau local |

---

## Phase 7 — Sécurité & Résilience

| Composant | TDD | Statut | Fichier | Écart |
|---|---|---|---|---|
| Réflexes spinaux (hard-coded, no STDP) | §19 | ✅ | `safety/reflexes.rs` | 5 types, override cognitif |
| BitFlipDetector (redondance topologique) | §15 | ✅ | `safety/hardware_resilience.rs` | Détection + correction ECC (32 blocs, syndrome, single-bit fix) |
| MemoryDiagnostics | §31 | ✅ | `safety/hardware_resilience.rs` | Ping mémoire + bad sector tracking |
| **Correction ECC / auto-réparation** | §15 | ✅ | hardware_resilience.rs | EccBlock: parity + syndrome + correction bit-flip, verify_all_ecc() |
| **Migration synaptique (secteur mort)** | §31 | ✅ | hardware_resilience.rs | migrate_synapse() copie vers nouvelle SynapseId, bad_sectors map |
| **Sénescence artificielle** | §31 | ✅ | hardware_resilience.rs | SenescenceStage (Healthy→Aging→Degraded→EndOfLife), score basé sur age |
| OTA (bank A/B, validation, rollback) | §16 | 16KB ✅ | `system/ota.rs` | Flash MMIO, CRC32, slot state machine, soft reset, persistence before switch, wiring in boot

---

## Phase 8 — Énergie & Métabolisme

| Composant | TDD | Statut | Fichier | Écart |
|---|---|---|---|---|
| PowerManager (5 modes) | §11, §34 | ✅ | `system/power.rs` | Active → Shutdown via WFI/Stop/Standby |
| Désactivation de couches par mode | §34 | ✅ | `system/power.rs` | Couches lentes → rapides |
| DVFS (5 OPP 16–216MHz) | §11 | ✅ | `system/power.rs` | PWR_CR VOS + OPP auto selon mode |
| Wake-up system (RTC, EXTI, TIM) | §34 | ✅ | `system/power.rs` | WakeUpConfig + check_wake_source() |
| Power budgeting par domaine (6) | §34 | ✅ | `system/power.rs` | CPU/Memory/Sensors/Actuators/Radio/Cognitive |
| Energy-Harvesting Aware | §34 | ✅ | `system/power.rs` | HarvestingType + MPPT perturb-and-observe |
| Mode Survie / Mode Exploration | §34 | ✅ | `system/power.rs` | Auto-switch selon batterie + harvesting |
| Couplage SNN → énergie | §11 | ✅ | `system/power.rs` + boot.rs | `net.energy_level = power_manager().battery_level` |
| Low-power idle adaptatif | §34 | ✅ | `system/power.rs` | `idle_if_possible()` → deep sleep si idle > 10ms |
| RCC clock gating par domaine | §34 | ✅ | `system/power.rs` | AHB1/APB1/APB2 ENR selon mode |
| **V_th dynamique (seuil selon énergie)** | §11 | ✅ | `snn/network.rs` | Piloté par `PowerManager::threshold_multiplier()` + battery_level |

---

## Phase 9 — Télémétrie & XAI

| Composant | TDD | Taille | Statut | Écart |
|---|---|---|---|---|
| SpikeTraceLogger (buffer circulaire) | §13 | 2.7KB | ⚠️ | Enregistrement ✅, pas d'export UART |
| CausalGraph (XAI) | §13 | 3.5KB | ⚠️ | Graphe causal complet 4096 edges, update avec spike_count + confiance EMÀ |
| **Export UART pour diagnostic externe** | §13 | xai.rs | ✅ | export_uart_text() → format texte structuré, 2048 bytes |
| **Reconstruction graphe causal visuel** | §13 | xai.rs | ⚠️ | top_causal_paths() trié par confiance |
| **Attribution de caractéristiques** | §13 | xai.rs | ✅ | FeatureAttribution avec contribution + signe, 128 slots |

---

## Phase 10 — Infrastructure & Qualité 📅

### 10.1 Tests

| Test | Statut |
|---|---|---|
| Tests unitaires core/math.rs | ✅ 38 tests |
| Tests unitaires core/memory.rs | ✅ 10 tests |
| Tests unitaires core/time.rs | ✅ 6 tests |
| Tests unitaires core/crypto.rs | ✅ 4 tests (2 nouveaux : non-aligned + multi-block) |
| Tests unitaires snn/neuron.rs | ✅ 8 tests |
| Tests unitaires snn/synapse.rs | ✅ 8 tests |
| Tests unitaires snn/plasticity.rs | ✅ 18 tests (11 nouveaux : calcium_model, plateau_potential) |
| Tests unitaires snn/neurogenesis.rs | ✅ 9 tests |
| Tests unitaires snn/network.rs | ✅ 14 tests |
| Tests unitaires io/buffers.rs | ✅ 5 tests |
| Tests unitaires io/isr.rs | ✅ 7 tests |
| Tests unitaires system/ota.rs | ✅ 11 tests |
| Tests unitaires system/power.rs | ✅ 29 tests |
| Tests unitaires cognitive/episodic.rs | ✅ 21 tests |
| Tests unitaires cognitive/attention.rs | ✅ 11 tests |
| Tests unitaires cognitive/curiosity.rs | ✅ 20 tests |
| Tests unitaires cognitive/networks.rs | ✅ 9 tests |
| Tests unitaires cognitive/reflex_override.rs | ✅ 4 tests |
| Tests unitaires safety/reflexes.rs | ✅ 8 tests |
| Tests unitaires safety/entropy_monitor.rs | ✅ 10 tests |
| Tests unitaires cognitive/proprioception.rs | ✅ 9 tests |
| Tests unitaires safety/hardware_resilience.rs | ✅ 10 tests |
| Tests unitaires swarm/mesh.rs | ✅ 12 tests |
| Tests unitaires swarm/federated.rs | ✅ 6 tests |
| Tests unitaires cognitive/neuromodulation.rs | ✅ 8 tests |
| Tests unitaires telemetry/spike_trace.rs | ✅ 13 tests |
| Tests unitaires telemetry/xai.rs | ✅ 16 tests (3 nouveaux : reconstruct_path_to + cycle) |
| Tests unitaires system/persistence.rs | ✅ 3 tests |
| Tests unitaires system/watchdog.rs | ✅ 5 tests |
| Tests unitaires bio/astrocytes.rs | ✅ 11 tests |
| Tests unitaires bio/striosome.rs | ✅ 13 tests |
| Tests unitaires bio/thalamus.rs | ✅ 12 tests |
| Tests unitaires bio/hippocampus.rs | ✅ 11 tests |
| Tests unitaires bio/cerebellum.rs | ✅ 12 tests |
| Tests unitaires cognitive/episodic.rs | ✅ 21 tests |
| Tests pipeline bio→cognitive | ✅ 7 tests (SWR bridge, spatial context, striosome, thalamus, astrocytes, cerebellum, full pipeline) |
| **Total** | **✅ 495 tests — tout vert** |
| Tests d'intégration | ✅ 13 tests (`tests/integration.rs` — 8 nouveaux : senescence, predictor, temporal, emergencies, network+cognitive, novelty, plasticity_100k, endurance) |
| Intégration réseau + cognitive | ✅ 2 tests : full_cycle (dopamine→SNN→calcium), novelty via predictor |
| **Stress test plasticité (100K cycles)** | ✅ `test_endurance_plasticity_100k` — STDP + calcium + plateau sans plantage |
| **Stress test endurance (1M cycles réseau)** | ✅ `endurance_million_cycles` — step() SNN 1M fois, time avance correctement |
| **Stress test pipeline (10K cycles)** | ✅ `endurance_stress_test_10k_cycles` — tous les modules bio + cognitifs, 10K itérations |
| **Stress test consolidation (5000 enregistrements)** | ✅ `endurance_consolidation_does_not_exhaust_memory` — 100 cycles record/consolidate, mémoire saine |
| **Benchmark HIL (Hardware-In-The-Loop)** | ❌ |
| **Test de résistance (5 ans simulés)** | ✅ `endurance_stress_test_10k_cycles` + time warp |
| **Banc FPGA (injection signaux)** | ❌ |

### 10.2 CI & Tooling

| Outil | Statut |
|---|---|---|
| GitHub Actions (check, test, clippy, fmt, cross, deny) | ✅ `.github/workflows/ci.yml` — 9 jobs : check, fmt, clippy, test, deny, cross-ARM, cross-RISCV |
 | Cross-compilation ARM Cortex-M7 (`thumbv7em-none-eabihf`) | ✅ CI + `.cargo/config.toml` — AtomicU64→AtomicU32, `write_volatile` import, `_reg` naming, unused vars |
| Cross-compilation RISC-V RV32 (`riscv32imac-unknown-none-elf`) | ✅ CI + `.cargo/config.toml` — `{0:e}`→`{0}` asm template |
| Cross-compilation RISC-V RV32-IMC (`riscv32imc-unknown-none-elf`) | ⚠️ Compile BSP uniquement (pas d'atomics hardware sur imc) |
| Vérification `cargo deny` (aucune dep) | ✅ `deny.toml` + CI |
| `.cargo/config.toml` (alias, rustflags) | ✅ `ct`, `cr`, `xt`, `xr` |
| **BSP STM32F7** (linker, startup) | ✅ | `stm32f746.ld`, `src/bsp/stm32f7.rs` | Cargo feature `stm32f7` |
| **BSP HiFive1** (linker, startup) | ✅ | `hifive1.ld`, `src/bsp/hifive1.rs` | Cargo feature `hifive1` |
| **BSP ESP32-C6** (RISC-V + WiFi) | ✅ | `esp32c6.ld`, `src/bsp/esp32c6.rs` | Cargo feature `esp32c6` |
| **Simulateur QEMU** | ✅ | `scripts/qemu_test.sh` — stm32f7, hifive1, esp32c6 | |
| **Custom Linker Script (.ld) → TCM** | ✅ | `stm32f746.ld` (+ hifive1, esp32c6 variants) | |

### 10.3 Qualité de code

| Métrique | Statut |
|---|---|
| `cargo clippy` — zéro warning | ✅ 0 warning |
| `cargo fmt` | ✅ Fait |
| `#![deny(clippy::all)]` | ✅ Ajouté dans `src/lib.rs` |
| Doc API complète (cargo doc) | ✅ |
| Documentation projet (README.md, LICENSE, .gitignore) | ✅ |
| Documentation technique (docs/ — 10 fichiers) | ✅ Architecture, Core, SNN, Cognitive, I/O, Swarm, Safety, System, Telemetry, Getting Started |
| `#![no_std]` respecté | ✅ |
| Zero-dependency | ✅ |

---

## Phase 11 — Vision Long Terme 🚀

### 11.1 Bio-inspiration avancée ✅ IMPLÉMENTÉ

| Concept | Fichier | Description | Statut |
|---|---|---|---|
| **Astrocytes** | `bio/astrocytes.rs` | Modulation gliale, ondes calciques lentes Ca²⁺, « troisième facteur » synaptique | ✅ 64 cellules, 4 nadirs, 11 tests |
| **Striosome/matrice** | `bio/striosome.rs` | Ganglions de la base : 16 striosomes + compartiments matrice, sélection d'action par dopamine | ✅ WTA + 13 tests |
| **Thalamus** | `bio/thalamus.rs` | 4 noyaux relais + TRN, gating sensoriel & bursting, routage attentionnel, 4 modes de décharge | ✅ 4 relais, 12 tests |
| **Hippocampe** | `bio/hippocampus.rs` | DG→CA3→CA1, séparation de patterns (256 granule → 128 CA3 → 128 CA1), SWR 50ms, détection de nouveauté | ✅ 11 tests, pont SWR → mémoire épisodique |
| **Cervelet** | `bio/cerebellum.rs` | 1024 Granulaires → 64 Purkinje, erreur CF, timing µs, apprentissage moteur fin | ✅ 12 tests |

**Intégration cross-module** : Tous les modules bio sont orchestrés depuis `snn/network.rs::process_bio_modules()` à leurs échelles de temps biologiques respectives (10–100ms). Pont SWR hippocampe → mémoire épisodique cognitive.

### 11.2 eFPGA — Bio-compilation (§30)

> *« Lorsqu'un groupe de neurones est identifié comme critique et immuable, l'orchestrateur génère un bitstream FPGA interne. Le temps d'exécution passe de la μs à la ns. »*

- [ ] Détection de motifs stables dans le réseau
- [ ] Génération de bitstream from software
- [ ] Reconfiguration dynamique du silicium
- [ ] Hardware-wiring des connexions critiques

### 11.3 Swarm intelligent distribué

- [ ] Essaim de 100+ nœuds
- [ ] Intelligence collective émergente
- [ ] Auto-réparation du réseau mesh
- [ ] Apprentissage fédéré hiérarchique

### 11.4 Apprentissage Continu (`src/cognitive/continual.rs`)

- [x] Consolidation offline par rejeu d'expériences (Sharp Wave-Ripples SWR 150-250 Hz)
- [x] Few-shot learning (adaptation rapide Fast-Weights)
- [x] Meta-learning (hyperparamètres adaptatifs $\eta_{\text{STDP}}$, $\theta_{\text{DA}}$)
- [x] Anti-oubli catastrophique validé (Elastic Weight Consolidation EWC & Matrice de Fisher $F_{ij}$)


---

## Phase 12 — Intelligence Visuelle, Spatiale & Physique Spiking (Option 2) 🌊

### 12.1 Traitement Visuel Rétinien & DVS
| Composant | Statut | Fichier | Description |
|---|---|---|---|
| **Retinal Engine (DoG)** | ✅ | `src/vision/retina.rs` | Filtre DoG $5\times5$ Center-Surround Q16.16 |
| **ON/OFF Ganglion Cells** | ✅ | `src/vision/retina.rs` | Dualité des voies de contraste |
| **DVS Event Polarity Encoder** | ✅ | `src/vision/retina.rs` | Encodage d'événements impulsionnels log-intensité ($+1$ / $-1$) |

### 12.2 Cortex Visuel V1 & V4
| Composant | Statut | Fichier | Description |
|---|---|---|---|
| **Gabor 2D Filter Bank** | ✅ | `src/vision/v1_gabor.rs` | Noyaux 0°, 45°, 90°, 135° pour contours V1 |
| **Visual Object Prototypes** | ✅ | `src/vision/v4_shape.rs` | Extraction de courbures et clustering Hebbien IT |

### 12.3 Cortex MT, Stéréovision & Intuitive Physics Engine
| Composant | Statut | Fichier | Description |
|---|---|---|---|
| **Reichardt EMD & Optical Flow** | ✅ | `src/vision/mt_motion.rs` | Vecteurs de vitesse $(V_x, V_y, V_z)$ et expansion/looming |
| **Stereo Depth Mapping** | ✅ | `src/vision/depth_spatial.rs` | Disparité stéréoscopique $Z = (f \cdot B) / d$ et cartes 3D |
| **Intuitive Physics Engine** | ✅ | `src/vision/physics_engine.rs` | Extrapolation ballistique $\vec{x}(t+\Delta t)$, gravité $g$, prédiction de collisions, et permanence sous occultation |
| **Predictive Coding & S-CNN** | ✅ | `src/vision/predictive_coding.rs`, `conv.rs` | Prédiction $I_{\text{pred}}$, erreur $\mathcal{E}_{\text{vis}}$, `SpikingConv2D`, `SpikingConv3D`, `SpikingMaxPool` |

---

## Phase 13 — Pont SNN ↔ LLM & Cognition Symbolique / NLP (Option 3) 💬

### 13.1 Tokenization & Encodage/Décodage Impulsionnel
| Composant | Statut | Fichier | Description |
|---|---|---|---|
| **Spike Tokenizer & Encoder** | ✅ | `src/nlp/spike_token.rs` | Tokenizer ASCII + BPE 256 avec encodage de position par phase temporelle |
| **Spike Token Decoder** | ✅ | `src/nlp/spike_decoder.rs` | Décodage WTA de la couche L4 et reconstruction de phrases |

### 13.2 Cognition Symbolique & Verbalisation
| Composant | Statut | Fichier | Description |
|---|---|---|---|
| **Neuromodulated Verbalizer** | ✅ | `src/nlp/verbalizer.rs` | Traduction en langage naturel des neuromodulateurs (DA, 5-HT, NA, ACh), erreur et curiosité |
| **Symbolic Knowledge Graph** | ✅ | `src/nlp/symbolic_graph.rs` | Graphe de concepts & triplets $(S, R, O)$ avec propagation d'activation Hebbienne |
| **Dialogue Engine** | ✅ | `src/nlp/dialogue_engine.rs` | Contrôleur unifié de dialogue NLP et d'explication d'état cognitif |

---

## Phase 14 — Intelligence Auditive & Vocalisation Spiking (Option 4) 🎙️

### 14.1 Cochlée & Cortex Auditif A1
| Composant | Statut | Fichier | Description |
|---|---|---|---|
| **Cochlea Gammatone Engine** | ✅ | `src/audio/cochlea.rs` | Banc de filtres 32 bandes ERB (80Hz..8000Hz) & PFM hair cells |
| **Cortex A1 & Formants** | ✅ | `src/audio/a1_formants.rs` | Carte tonotopique A1, formants ($F_1, F_2, F_3$) & voyelles |

### 14.2 Prosodie & Synthèse Vocale
| Composant | Statut | Fichier | Description |
|---|---|---|---|
| **Pitch F0 & Onset Engine** | ✅ | `src/audio/pitch_rhythm.rs` | Autocorrelation $F_0$ (voix homme/femme) & détection de rythme |
| **Spike Voice Synthesizer** | ✅ | `src/audio/voice_synth.rs` | Synthétiseur vocal résonant produisant du PCM 16-bit 16kHz |

---

## Phase 15 — Bio-Compilation eFPGA & Accélération Silicium (Option 6) ⚡

### 15.1 Analyse de Stabilité & Génération Verilog
| Composant | Statut | Fichier | Description |
|---|---|---|---|
| **Subnetwork Stability Analyzer** | ✅ | `src/efpga/stability.rs` | Calcul de variance $\sigma_w^2$ & figement des sous-réseaux immuables |
| **Synthesizable Verilog HDL Generator** | ✅ | `src/efpga/hdl_gen.rs` | Génération de modules Verilog RTL (`module efpga_snn_subnetwork`) |

### 15.2 Bitstream eFPGA & Simulation Sub-Nanoseconde
| Composant | Statut | Fichier | Description |
|---|---|---|---|
| **eFPGA Bitstream Configurator** | ✅ | `src/efpga/bitstream.rs` | Compilateur de bitstream binaire pour LUT4/LUT6 & matrices de routage |
| **Nanosecond Hardware Simulator** | ✅ | `src/efpga/simulator.rs` | Simulation cycle-accurate (<1 ns par spike, accélération >1000x) |

---

## Résumé par module





```
MODULE           TAILLE   TDD   STATUT    RESTANT
core/math        13.9KB   —     ✅ 100%   —
core/memory      17.2KB   —     ✅ 100%   —
core/time        11.1KB   §18   ✅ 100%   —
core/entropy       14.0KB   §32   ✅ 100%   Seuils adaptatifs (mean±2σ) + CognitiveMode → NM
core/crypto       8.9KB   §10   ✅ 100%   —
snn/neuron       13.9KB   §3.1  ✅ 100%   —
snn/synapse      15.0KB   §5.3  ✅ 100%   apply_senescence(), init_reflex_arcs() L0→L6→L4, apply_plasticity_modulation() non cumulée
snn/network      18.3KB   §6    ✅ 100%   Cycle prédictif connecté via `predictive_cycle()`
snn/plasticity    12.0KB  §5.3  ✅ 100%   CalciumModel + PlateauPotential implémentés, gating STDP, 18 tests
snn/homeostasis   4.0KB   §20   ✅ 100%   —
snn/neurogenesis  8.1KB   §12   ✅ 100%   apply_senescence(), max_age configurable, maintenance_cycle() retourne (pruned, created, senesced)
io/buffers        7.0KB   §4    ✅ 100%   —
io/isr            13.0KB  §23   ✅ 100%   NVIC + handlers TIM2/ADC/EXTI/SPI/I2C → GLOBAL_SPIKE_QUEUE
io/encoder        9.8KB   §4    ✅ 100%   —
io/decoder        2.7KB   §28   ✅ 100%   —
io/sensors       16.0KB   §4    ✅ 100%   I2C/SPI/ADC MMIO, SensorManager global

io/actuators      9.2KB   §28   ✅ 100%   PWM/GPIO/DAC register writes, ActuatorManager global
cognitive/actor               4.5KB   §6    ✅ 100%   Actor-Critic RL complet (Policy, Value, TD-error, dopamine)
cognitive/attention           5.3KB   §14   ⚠️  90%   WTA ✅, saliency map ✅, top-down routing ✅, focus gain/suppression ✅
cognitive/curiosity           6.8KB   §29   ⚠️  85%   Bruit ✅, habituation ✅, ennui ✅, seuils adaptatifs ✅, bruit thermique → sensory layer ✅, ε-greedy adaptatif ✅
cognitive/neuromodulation     2.4KB   §21   ✅ 100%   Boucle apprentissage ✅, calibration auto ✅
cognitive/networks            2.6KB   §21   ✅ 100%   NeuromodulationCalibration (EMA, volatilité, decay adaptatif, sensitivity)
cognitive/predictor          16.0KB   §6    ✅ 100%   Prototypes + Hebbian online learning, confidence, transition buffer 256
cognitive/proprioception      5.2KB   §22   ⚠️  90%   Erreur ✅, correction ✅, modèle corporel ✅
cognitive/temporal          12.0KB   §33   ✅ 100%   Time cells 64 offsets + sequence buffer + interval timing

swarm/federated               2.5KB   §8    ✅ 100%   DP ✅, topologie adaptative ✅, fiabilité ✅
swarm/mesh                    5.2KB   §8    ✅ 100%   Δw ✅, discovery ✅, gossip ✅, clock sync ✅, remote spikes ✅

safety/reflexes               4.8KB   §19   ✅  100%   Règles + override cognitif + 12 tests
cognitive/reflex_override     1.5KB   §19   ✅  100%   Évaluation NA + mode + attention, 4 tests
safety/entropy_monitor        3.5KB   §32   ✅  100%   Seuils adaptatifs + 10 tests
safety/hardware_resilience    3.8KB   §15   ✅ 100%   ECC ✅, migration synaptique ✅, sénescence ✅

system/boot                   7.2KB   §9    ✅ 100%   init_hardware_peripherals() (CPACR/SCB_VTOR/MPU/CCR), read_boot_config() (UID OTP), enable_sensor_interrupts() (NVIC_ISER), check_emergencies() (reflexes/entropy/NEURON_COUNT), cycle OTA t=21ms
system/persistence           11.0KB   §7    ✅ 100%   commit_to_flash() STM32F7 MMIO, rotation J-0/J-1/J-2, 3 tests
system/watchdog               2.8KB   §7.2  ✅ 100%   NeutronicWatchdog, 5 tests
system/power                  24KB    §11   ✅ 100%   DVFS, wake-up, budget, harvest, V_th piloté par PowerManager ✅
system/ota                    16KB    §16   ✅ 100%   Dual-bank, CRC32, rollback, flash MMIO, soft reset, no_std (stack buffer [u8;1028] au lieu de Vec)

system/watchdog               2.8KB   §7.2  ✅ 100%   NeutronicWatchdog, 5 tests
system/power                  24KB    §11   ✅ 100%   DVFS, wake-up, budget, harvest, V_th piloté par PowerManager ✅
system/ota                    16KB    §16   ✅ 100%   Dual-bank, CRC32, rollback, flash MMIO, soft reset, no_std (stack buffer [u8;1028] au lieu de Vec)

telemetry/spike_trace         5.5KB   §13   ✅ 100%   Buffer + export UART + 13 tests
telemetry/xai                 5.0KB   §13   ✅  85%   CausalGraph global, reconstruct_path_to(), export UART, 15 tests

[HKL-2 MODULES — --features hkl2]
learning/surrogate            2.8KB   —     ✅ 100%   Fast Sigmoid, ArcTan, Straight-Through fixed-point
learning/eprop                3.0KB   —     ✅ 100%   Eligibility propagation, trace decay, weight delta
learning/loss                 2.2KB   —     ✅ 100%   Spiking cross-entropy loss & learning signals
embedding/spike_embedding     3.6KB   —     ✅ 100%   SpikeEmbeddingLayer 256D, LIFNeuronLight, temporal coding
embedding/bpe_tokenizer       3.4KB   —     ✅ 100%   BPE Byte-Pair Encoding & decoding
transformer/norm              2.2KB   —     ✅ 100%   FixedPoint LayerNorm
transformer/attention         6.7KB   —     ✅ 100%   Spiking Self-Attention 4-head Softmax-free
transformer/feed_forward      3.4KB   —     ✅ 100%   Spiking FFN 256->512->256
transformer/block             2.3KB   —     ✅ 100%   SpikingTransformerBlock avec LayerNorm & résiduels
transformer/backbone          4.5KB   —     ✅ 100%   SpikingTransformer model + OutputProjection 4096-vocab
training/data_loader          1.8KB   —     ✅ 100%   TextDataLoader (fenêtre glissante)
training/trainer              3.0KB   —     ✅ 100%   End-to-end Trainer pipeline (e-prop + loss)
```

---

## Phase HKL-2 — Spiking Foundation Model (Architecture Cerveau Artificiel) ✅

> **Paradigme HKL-2** : Évolution du noyau SNN embarqué vers un Modèle de Fondation à Spikes capable d'apprentissage global (e-prop), de représentations distribuées (population coding), et de raisonnement séquentiel (Spiking Transformer).

| Composant | Statut | Fichier | Description |
|---|---|---|---|
| **Corrections Critiques Phase 0** | ✅ | `audio/cochlea.rs`, `audio/voice_synth.rs`, `vision/depth_spatial.rs` | Cochlée I/Q quadrature, Synthèse vocale IIR Biquad formants, Stéréo SAD 5×5 |
| **Gradients Surrogate** | ✅ | `learning/surrogate.rs` | Fast Sigmoid, ArcTan, Straight-Through en virgule fixe Q16.16 |
| **e-prop Engine** | ✅ | `learning/eprop.rs` | Eligibility propagation online ($e_{ij}(t) = \alpha e_{ij}(t-1) + \text{surrogate}(U_j) \text{spike}_i$) |
| **Loss Spiking Cross-Entropy** | ✅ | `learning/loss.rs` | Calcul de perte & signaux d'erreur descendants $L_j$ |
| **Population Spike Embedding** | ✅ | `embedding/spike_embedding.rs` | 256D spatio-temporel sur $T=4$ pas de temps (`LIFNeuronLight`) |
| **BPE Tokenizer** | ✅ | `embedding/bpe_tokenizer.rs` | Tokenizer Byte-Pair Encoding avec fusion de paires |
| **Spiking Self-Attention (SSA)** | ✅ | `transformer/attention.rs` | Attention 4 têtes sans Softmax sur flux de spikes binaires Q/K/V |
| **Spiking Feed-Forward (FFN)** | ✅ | `transformer/feed_forward.rs` | MLP $256 \to 512 \to 256$ en neurones à spikes |
| **Spiking Transformer Block** | ✅ | `transformer/block.rs` | Bloc résiduel avec dual LayerNorm |
| **Spiking Transformer Backbone** | ✅ | `transformer/backbone.rs` | Modèle $N$-couches avec tête `OutputProjection` (vocab 4096) |
| **Trainer & Data Loader** | ✅ | `training/data_loader.rs`, `training/trainer.rs` | Ingestion autoregressive, calcul de perte & maj e-prop |

---

## Priorités recommandées

### 🔴 Immédiat (Phase 3 → Phase 4)

Ces modules sont le cœur du TDD et doivent être implémentés pour que HKL-1 tienne ses promesses conceptuelles.

| Priorité | Tâche | Modules | Effort |
|---|---|---|---|
| P0 | **Cycle prédictif complet** (Acteur ↔ Prédicteur ↔ TimeWarper) | ✅ FAIT |
| P0 | **Actor-Critic RL** (Policy, Value, TD-error, dopamine) | ✅ FAIT |
| P1 | **Interruptions matérielles (ISR) → RingBuffer** | ✅ FAIT |
| P1 | **Drivers sensors I2C/SPI/ADC** | ✅ FAIT |
| P1 | **Drivers actuators PWM/GPIO** | ✅ FAIT |

### 🟡 Court terme (Phase 5 → Phase 8)

| Priorité | Tâche | Effort |
|---|---|---|
| P2 | Predictor network (forward dynamics) | ✅ FAIT |
| P2 | Temporal cognition (time cells, sequences) | ✅ FAIT |
| P2 | Entropie cognitive → boucle neuromodulation | ✅ FAIT |
| P2 | OTA complet (bank A/B, validation) | ✅ FAIT |
| P2 | Custom Linker Script (.ld) pour TCM | ✅ FAIT |
| P2 | BSP STM32F7 (linker + startup) | ✅ FAIT |
| P2 | BSP HiFive1 (linker + startup) | ✅ FAIT |
| P2 | BSP ESP32-C6 (linker + startup) | ✅ FAIT |
| P3 | Override cognitif des réflexes | ✅ FAIT |
| P3 | Seuils adaptatifs entropy_monitor | ✅ FAIT |

### 🟢 Long terme (Phase 9 → Phase 11)

| Priorité | Tâche | Effort |
|---|---|---|
| P4 | Tests unitaires et d'intégration | ✅ FAIT — 753 tests (2 stress tests) |
| P4 | CI/CD (GitHub Actions, cross-compilation) | ✅ FAIT (`.github/workflows/ci.yml` + `.cargo/config.toml`) |
| P4 | BSP cibles matérielles (STM32F7, RISC-V, ESP32) | ✅ FAIT (linker + startup + check_emergencies) |
| P5 | Sénescence synaptique + migration | ✅ FAIT (senescence + apply_senescence + init_reflex_arcs) |
| P5 | HIL test bench (QEMU hardware-in-loop) | ~1 mois |
| P6 | eFPGA bio-compilation | ✅ FAIT (`src/efpga/`) |
| P6 | Astrocytes, thalamus, hippocampe | ✅ FAIT (`src/bio/`) |
| P7 | Entraînement HKL-2 sur corpus de texte | ~1 mois |

