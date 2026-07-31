use crate::core::math::FixedPoint;
use crate::core::memory::{NEURON_COUNT, NeuronId, neuron_state, neuron_state_ref};

const MAX_LAYERS: u8 = 8;
const SALIENCY_HISTORY: usize = 16;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FocusType {
    None,
    BottomUpSalient,
    GoalDriven,
    Exploratory,
}

#[derive(Clone, Copy)]
pub struct SaliencyLayer {
    pub bottom_up: FixedPoint,
    pub top_down_bias: FixedPoint,
    pub combined: FixedPoint,
    pub peak_neuron: u16,
    pub peak_value: FixedPoint,
}

pub struct SaliencyMap {
    pub layers: [SaliencyLayer; 8],
    pub most_salient_layer: u8,
    pub history: [FixedPoint; SALIENCY_HISTORY],
    pub history_idx: u8,
    pub history_count: u8,
    pub prediction_error_influence: FixedPoint,
    pub novelty_influence: FixedPoint,
    pub goal_bias_influence: FixedPoint,
}

impl SaliencyMap {
    pub const fn new() -> Self {
        Self {
            layers: [SaliencyLayer {
                bottom_up: FixedPoint::ZERO,
                top_down_bias: FixedPoint::ZERO,
                combined: FixedPoint::ZERO,
                peak_neuron: 0,
                peak_value: FixedPoint::ZERO,
            }; 8],
            most_salient_layer: 0,
            history: [FixedPoint::ZERO; SALIENCY_HISTORY],
            history_idx: 0,
            history_count: 0,
            prediction_error_influence: FixedPoint::from_f32(0.4),
            novelty_influence: FixedPoint::from_f32(0.3),
            goal_bias_influence: FixedPoint::from_f32(0.3),
        }
    }

    pub fn recompute_bottom_up(
        &mut self,
        prediction_error: FixedPoint,
        novelty: FixedPoint,
        goal_bias: FixedPoint,
    ) {
        let count = NEURON_COUNT.load(core::sync::atomic::Ordering::Relaxed);
        if count == 0 {
            return;
        }

        for l in 0..MAX_LAYERS {
            self.layers[l as usize].bottom_up = FixedPoint::ZERO;
            self.layers[l as usize].peak_neuron = 0;
            self.layers[l as usize].peak_value = FixedPoint::ZERO;
        }

        let mut total_firing = FixedPoint::ZERO;
        for i in 0..count as u16 {
            let id = NeuronId::new(i);
            let state = neuron_state_ref(id);
            let layer = state.layer;
            if layer >= MAX_LAYERS {
                continue;
            }
            let potential = state.membrane_potential.abs();
            self.layers[layer as usize].bottom_up += potential;
            if potential > self.layers[layer as usize].peak_value {
                self.layers[layer as usize].peak_value = potential;
                self.layers[layer as usize].peak_neuron = i;
            }
            total_firing += potential;
        }

        for l in 0..MAX_LAYERS as usize {
            let buf = &mut self.layers[l];
            if total_firing > FixedPoint::ZERO {
                let norm = buf.bottom_up / total_firing;
                buf.bottom_up = norm * FixedPoint::from_f32(0.5)
                    + prediction_error * self.prediction_error_influence
                    + novelty * self.novelty_influence
                    + goal_bias * self.goal_bias_influence;
            }
            buf.combined = buf.bottom_up + buf.top_down_bias;
        }

        let mut best = FixedPoint::ZERO;
        for l in 0..MAX_LAYERS as usize {
            if self.layers[l].combined > best {
                best = self.layers[l].combined;
                self.most_salient_layer = l as u8;
            }
        }

        if self.history_count < SALIENCY_HISTORY as u8 {
            self.history_count += 1;
        }
        self.history[self.history_idx as usize] =
            self.layers[self.most_salient_layer as usize].combined;
        self.history_idx = (self.history_idx + 1) % (SALIENCY_HISTORY as u8);
    }

    pub fn set_top_down_bias(&mut self, target_layer: u8, bias: FixedPoint) {
        if target_layer < MAX_LAYERS {
            self.layers[target_layer as usize].top_down_bias = bias;
        }
        for l in 0..MAX_LAYERS as usize {
            if l != target_layer as usize {
                self.layers[l].top_down_bias = FixedPoint::ZERO;
            }
        }
    }

    pub fn average_saliency(&self) -> FixedPoint {
        if self.history_count == 0 {
            return FixedPoint::ZERO;
        }
        let mut sum = FixedPoint::ZERO;
        let count = self.history_count.min(SALIENCY_HISTORY as u8) as usize;
        for i in 0..count {
            sum += self.history[i];
        }
        sum / FixedPoint::from_int(count as i32)
    }
}

pub struct AttentionFocus {
    pub focus_type: FocusType,
    pub target_layer: u8,
    pub target_neuron: u16,
    pub spread: u16,
    pub gain: FixedPoint,
    pub suppression: FixedPoint,
    pub dwell_counter: u32,
    pub dwell_target: u32,
    pub shifted_recently: bool,
}

impl AttentionFocus {
    pub const fn new() -> Self {
        Self {
            focus_type: FocusType::None,
            target_layer: 0,
            target_neuron: 0,
            spread: 32,
            gain: FixedPoint::from_f32(1.2),
            suppression: FixedPoint::from_f32(0.8),
            dwell_counter: 0,
            dwell_target: 100,
            shifted_recently: false,
        }
    }

    pub fn set_focus(&mut self, ftype: FocusType, layer: u8, neuron: u16) {
        self.focus_type = ftype;
        self.target_layer = layer;
        self.target_neuron = neuron;
        self.dwell_counter = 0;
        self.shifted_recently = true;
    }

    pub fn update(&mut self) {
        if self.dwell_counter < self.dwell_target {
            self.dwell_counter += 1;
            self.shifted_recently = false;
        }
    }

    pub fn is_attended(&self, neuron_id: NeuronId, layer: u8) -> bool {
        if self.focus_type == FocusType::None {
            return true;
        }
        if layer != self.target_layer {
            return false;
        }
        let idx = neuron_id.index() as u16;
        let half = self.spread / 2;
        let low = self.target_neuron.saturating_sub(half);
        let high = self.target_neuron.saturating_add(half);
        idx >= low && idx <= high
    }

    pub fn focus_shift_interval(&self) -> u32 {
        let na =
            unsafe { &crate::cognitive::neuromodulation::COGNITIVE_NEUROMODULATORS }.noradrenaline;
        let interval = (200.0 - na.to_f32() * 150.0) as u32;
        if interval < 50 {
            50
        } else if interval > 400 {
            400
        } else {
            interval
        }
    }
}

pub struct AttentionRouter {
    pub saliency_map: SaliencyMap,
    pub focus: AttentionFocus,
    pub bottom_up_weight: FixedPoint,
    pub top_down_weight: FixedPoint,
    pub goal_bias: FixedPoint,
    pub last_action_mapped_layer: u8,
    pub update_counter: u32,
}

impl AttentionRouter {
    pub fn new() -> Self {
        Self {
            saliency_map: SaliencyMap::new(),
            focus: AttentionFocus::new(),
            bottom_up_weight: FixedPoint::from_f32(0.5),
            top_down_weight: FixedPoint::from_f32(0.5),
            goal_bias: FixedPoint::from_f32(0.3),
            last_action_mapped_layer: 2,
            update_counter: 0,
        }
    }

    fn map_action_to_layer(action: u8) -> u8 {
        match action {
            0..=31 => 2,
            32..=63 => 3,
            64..=95 => 1,
            96..=127 => 0,
            128..=159 => 5,
            160..=191 => 6,
            _ => 4,
        }
    }

    pub fn update(
        &mut self,
        prediction_error: FixedPoint,
        novelty: FixedPoint,
        selected_action: Option<u8>,
        action_confidence: FixedPoint,
    ) {
        self.update_counter += 1;

        if self.update_counter % 5 == 0 {
            self.saliency_map
                .recompute_bottom_up(prediction_error, novelty, self.goal_bias);
        }

        if let Some(action) = selected_action {
            let target_layer = Self::map_action_to_layer(action);
            self.last_action_mapped_layer = target_layer;
            let bias = action_confidence * self.top_down_weight;
            self.saliency_map.set_top_down_bias(target_layer, bias);
        }

        let saliency_target = self.saliency_map.most_salient_layer;
        let peak_neuron = self.saliency_map.layers[saliency_target as usize].peak_neuron;

        let shift_interval = self.focus.focus_shift_interval();

        if self.update_counter % shift_interval == 0 && self.focus.focus_type != FocusType::None {
            let mut use_layer = saliency_target;
            if let Some(action) = selected_action {
                let goal_layer = Self::map_action_to_layer(action);
                if self.top_down_weight > self.bottom_up_weight {
                    use_layer = goal_layer;
                }
            }
            self.focus
                .set_focus(FocusType::BottomUpSalient, use_layer, peak_neuron);
        } else if self.focus.focus_type == FocusType::None {
            self.focus
                .set_focus(FocusType::BottomUpSalient, saliency_target, peak_neuron);
        }

        self.focus.update();
    }

    pub fn apply(&self) {
        let focus = &self.focus;
        if focus.focus_type == FocusType::None {
            return;
        }

        let ach =
            unsafe { &crate::cognitive::neuromodulation::COGNITIVE_NEUROMODULATORS }.acetylcholine;
        let dynamic_gain = FixedPoint::from_f32(1.0 + (0.2 + ach.to_f32() * 0.3));
        let dynamic_suppression = FixedPoint::from_f32(0.8 - ach.to_f32() * 0.2);

        let count = NEURON_COUNT.load(core::sync::atomic::Ordering::Relaxed);
        for i in 0..count as u16 {
            let id = NeuronId::new(i);
            let state = neuron_state_ref(id);
            let layer = state.layer;
            if focus.is_attended(id, layer) {
                let boost = neuron_state(id);
                boost.membrane_potential = boost.membrane_potential * dynamic_gain;
            } else {
                let suppress = neuron_state(id);
                suppress.membrane_potential = suppress.membrane_potential * dynamic_suppression;
            }
        }

        self.apply_wta_inhibition();
    }

    pub fn apply_wta_inhibition(&self) {
        let spread_half = self.focus.spread / 2;
        let count = NEURON_COUNT.load(core::sync::atomic::Ordering::Relaxed) as u16;
        if count == 0 {
            return;
        }
        for l in 0..8 {
            let peak_neuron = self.saliency_map.layers[l].peak_neuron;
            let low = peak_neuron.saturating_sub(spread_half);
            let high = peak_neuron
                .saturating_add(spread_half)
                .min(count.saturating_sub(1));
            for idx in low..=high {
                if idx == peak_neuron {
                    continue;
                }
                unsafe {
                    let neuron = crate::core::memory::NEURON_ARRAY[idx as usize].assume_init_mut();
                    if neuron.layer == l as u8 {
                        neuron.membrane_potential =
                            neuron.membrane_potential * FixedPoint::from_f32(0.7);
                    }
                }
            }
        }
    }

    pub fn apply_top_down_bias(&mut self, layer: u8, bias: FixedPoint) {
        self.saliency_map.set_top_down_bias(layer, bias);
        let count = NEURON_COUNT.load(core::sync::atomic::Ordering::Relaxed);
        for i in 0..count as u16 {
            let id = NeuronId::new(i);
            let state = neuron_state_ref(id);
            if state.layer == layer {
                neuron_state(id).bias_current += bias * FixedPoint::from_f32(0.1);
            }
        }
    }

    pub fn shift_to_exploratory(&mut self, rng: &mut crate::core::math::XorShift64Star) {
        let layer = (rng.next_u32() % (MAX_LAYERS as u32)) as u8;
        let count = NEURON_COUNT.load(core::sync::atomic::Ordering::Relaxed);
        let neuron = if count > 0 {
            (rng.next_u32() % count as u32) as u16
        } else {
            0
        };
        self.focus.set_focus(FocusType::Exploratory, layer, neuron);
    }
}

impl Default for AttentionRouter {
    fn default() -> Self {
        Self::new()
    }
}

use core::mem::MaybeUninit;
pub static mut ATTENTION_ROUTER: MaybeUninit<AttentionRouter> = MaybeUninit::uninit();

static INITIALIZED_ATTENTION_ROUTER: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

pub fn init_attention_router() {
    unsafe {
        ATTENTION_ROUTER.write(AttentionRouter::new());
        INITIALIZED_ATTENTION_ROUTER.store(true, core::sync::atomic::Ordering::Relaxed);
    }
}

pub fn attention_router() -> &'static mut AttentionRouter {
    unsafe {
        if !INITIALIZED_ATTENTION_ROUTER.load(core::sync::atomic::Ordering::Relaxed) {
            init_attention_router();
        }
        &mut *ATTENTION_ROUTER.as_mut_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_saliency_map_new() {
        let sm = SaliencyMap::new();
        for l in 0..8 {
            assert_eq!(sm.layers[l].bottom_up, FixedPoint::ZERO);
            assert_eq!(sm.layers[l].top_down_bias, FixedPoint::ZERO);
            assert_eq!(sm.layers[l].combined, FixedPoint::ZERO);
        }
        assert_eq!(sm.most_salient_layer, 0);
    }

    #[test]
    fn test_saliency_map_set_top_down_bias() {
        let mut sm = SaliencyMap::new();
        sm.set_top_down_bias(3, FixedPoint::from_f32(0.8));
        assert_eq!(sm.layers[3].top_down_bias, FixedPoint::from_f32(0.8));
        for l in 0..8 {
            if l != 3 {
                assert_eq!(sm.layers[l].top_down_bias, FixedPoint::ZERO);
            }
        }
    }

    #[test]
    fn test_attention_focus_new() {
        let af = AttentionFocus::new();
        assert_eq!(af.focus_type, FocusType::None);
        assert_eq!(af.gain, FixedPoint::from_f32(1.2));
        assert_eq!(af.suppression, FixedPoint::from_f32(0.8));
    }

    #[test]
    fn test_attention_focus_set() {
        let mut af = AttentionFocus::new();
        af.set_focus(FocusType::GoalDriven, 2, 100);
        assert_eq!(af.focus_type, FocusType::GoalDriven);
        assert_eq!(af.target_layer, 2);
        assert_eq!(af.target_neuron, 100);
        assert!(af.shifted_recently);
    }

    #[test]
    fn test_is_attended_none() {
        let af = AttentionFocus::new();
        assert!(af.is_attended(NeuronId::new(0), 0));
        assert!(af.is_attended(NeuronId::new(500), 7));
    }

    #[test]
    fn test_is_attended_with_focus() {
        let mut af = AttentionFocus::new();
        af.set_focus(FocusType::GoalDriven, 2, 100);
        af.spread = 32;
        assert!(af.is_attended(NeuronId::new(100), 2));
        assert!(af.is_attended(NeuronId::new(84), 2));
        assert!(af.is_attended(NeuronId::new(116), 2));
        assert!(!af.is_attended(NeuronId::new(100), 1));
        assert!(!af.is_attended(NeuronId::new(50), 2));
    }

    #[test]
    fn test_map_action_to_layer() {
        assert_eq!(AttentionRouter::map_action_to_layer(0), 2);
        assert_eq!(AttentionRouter::map_action_to_layer(64), 1);
        assert_eq!(AttentionRouter::map_action_to_layer(96), 0);
        assert_eq!(AttentionRouter::map_action_to_layer(128), 5);
        assert_eq!(AttentionRouter::map_action_to_layer(200), 4);
    }

    #[test]
    fn test_attention_router_new() {
        let ar = AttentionRouter::new();
        assert_eq!(ar.bottom_up_weight, FixedPoint::from_f32(0.5));
        assert_eq!(ar.top_down_weight, FixedPoint::from_f32(0.5));
    }

    #[test]
    fn test_attention_router_update() {
        let mut ar = AttentionRouter::new();
        ar.update(
            FixedPoint::from_f32(0.1),
            FixedPoint::from_f32(0.05),
            Some(64),
            FixedPoint::from_f32(0.7),
        );
        assert!(ar.update_counter > 0);
        assert_eq!(ar.last_action_mapped_layer, 1);
    }

    #[test]
    fn test_attention_focus_dwell() {
        let mut af = AttentionFocus::new();
        af.set_focus(FocusType::BottomUpSalient, 1, 50);
        for _ in 0..50 {
            af.update();
        }
        assert_eq!(af.dwell_counter, 50);
        assert!(!af.shifted_recently);
    }

    #[test]
    fn test_saliency_average_empty() {
        let sm = SaliencyMap::new();
        assert_eq!(sm.average_saliency(), FixedPoint::ZERO);
    }

    #[test]
    fn test_wta_lateral_inhibition() {
        let mut ar = AttentionRouter::new();
        ar.focus.set_focus(FocusType::BottomUpSalient, 0, 10);
        ar.saliency_map.layers[0].peak_neuron = 10;
        unsafe {
            crate::core::memory::NEURON_COUNT.store(20, core::sync::atomic::Ordering::Relaxed);
            for i in 0..20 {
                let mut n = crate::core::memory::NeuronState::default();
                n.membrane_potential = FixedPoint::from_f32(1.0);
                crate::core::memory::NEURON_ARRAY[i].write(n);
            }
        }
        ar.apply_wta_inhibition();
        unsafe {
            let n10 = crate::core::memory::NEURON_ARRAY[10].assume_init_mut();
            assert_eq!(n10.membrane_potential, FixedPoint::from_f32(1.0));
            let n9 = crate::core::memory::NEURON_ARRAY[9].assume_init_mut();
            assert_eq!(n9.membrane_potential, FixedPoint::from_f32(0.7));
        }
    }

    #[test]
    fn test_dynamic_focus_shift_interval() {
        let af = AttentionFocus::new();
        unsafe {
            crate::cognitive::neuromodulation::COGNITIVE_NEUROMODULATORS.noradrenaline =
                FixedPoint::from_f32(1.0);
        }
        assert_eq!(af.focus_shift_interval(), 50);
        unsafe {
            crate::cognitive::neuromodulation::COGNITIVE_NEUROMODULATORS.noradrenaline =
                FixedPoint::from_f32(0.0);
        }
        assert_eq!(af.focus_shift_interval(), 200);
    }

    #[test]
    fn test_ach_modulates_gain() {
        let mut ar = AttentionRouter::new();
        ar.focus.set_focus(FocusType::GoalDriven, 0, 0);
        unsafe {
            crate::core::memory::NEURON_COUNT.store(1, core::sync::atomic::Ordering::Relaxed);
            let mut n = crate::core::memory::NeuronState::default();
            n.membrane_potential = FixedPoint::from_f32(1.0);
            n.layer = 0;
            crate::core::memory::NEURON_ARRAY[0].write(n);
            crate::cognitive::neuromodulation::COGNITIVE_NEUROMODULATORS.acetylcholine =
                FixedPoint::from_f32(1.0);
        }
        ar.apply();
        unsafe {
            let n0 = crate::core::memory::NEURON_ARRAY[0].assume_init_mut();
            assert_eq!(n0.membrane_potential, FixedPoint::from_f32(1.5));
        }
    }
}
