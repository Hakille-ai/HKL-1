use crate::core::math::FixedPoint;
use crate::core::memory::NeuronId;

pub const MAX_EDGES: usize = 4096;
pub const MAX_FEATURES: usize = 128;

#[derive(Clone, Copy)]
pub struct CausalEdge {
    pub pre: NeuronId,
    pub post: NeuronId,
    pub weight: FixedPoint,
    pub confidence: FixedPoint,
    pub spike_count: u32,
    pub avg_latency: u32,
    pub layer_pair: (u8, u8),
}

impl CausalEdge {
    pub const fn empty() -> Self {
        Self {
            pre: NeuronId::INVALID,
            post: NeuronId::INVALID,
            weight: FixedPoint::ZERO,
            confidence: FixedPoint::ZERO,
            spike_count: 0,
            avg_latency: 0,
            layer_pair: (0, 0),
        }
    }
}

#[derive(Clone, Copy)]
pub struct FeatureAttribution {
    pub feature_idx: u16,
    pub contribution: FixedPoint,
    pub sign: i8,
    pub description: [u8; 32],
}

impl FeatureAttribution {
    pub const fn empty() -> Self {
        Self {
            feature_idx: 0,
            contribution: FixedPoint::ZERO,
            sign: 0,
            description: [0; 32],
        }
    }
}

pub struct CausalGraph {
    pub edges: [CausalEdge; MAX_EDGES],
    pub edge_count: u16,
    pub features: [FeatureAttribution; MAX_FEATURES],
    pub feature_count: u16,
    pub graph_density: FixedPoint,
    pub avg_confidence: FixedPoint,
    pub analysis_count: u32,
}

impl CausalGraph {
    pub fn new() -> Self {
        Self {
            edges: [CausalEdge::empty(); MAX_EDGES],
            edge_count: 0,
            features: [FeatureAttribution::empty(); MAX_FEATURES],
            feature_count: 0,
            graph_density: FixedPoint::ZERO,
            avg_confidence: FixedPoint::ZERO,
            analysis_count: 0,
        }
    }

    pub fn add_edge(&mut self, pre: NeuronId, post: NeuronId, weight: FixedPoint, latency: u32) {
        if self.edge_count as usize >= MAX_EDGES {
            return;
        }
        let idx = self.edge_count as usize;
        self.edges[idx] = CausalEdge {
            pre,
            post,
            weight,
            confidence: FixedPoint::from_f32(0.5),
            spike_count: 1,
            avg_latency: latency,
            layer_pair: (0, 0),
        };
        self.edge_count += 1;
        self.update_metrics();
    }

    pub fn update_edge(&mut self, pre: NeuronId, post: NeuronId, latency: u32) {
        for i in 0..self.edge_count as usize {
            let e = &mut self.edges[i];
            if e.pre == pre && e.post == post {
                e.spike_count += 1;
                let new_lat = e.avg_latency as f32 * 0.9 + latency as f32 * 0.1;
                e.avg_latency = new_lat as u32;
                let conf = e.spike_count as f32 / (e.spike_count as f32 + 10.0);
                e.confidence = FixedPoint::from_f32(conf);
                self.update_metrics();
                return;
            }
        }
        let pre_state = crate::core::memory::neuron_state_ref(pre);
        let post_state = crate::core::memory::neuron_state_ref(post);
        self.add_edge(pre, post, FixedPoint::ZERO, latency);
        let last_idx = self.edge_count as usize - 1;
        self.edges[last_idx].layer_pair = (pre_state.layer, post_state.layer);
    }

    pub fn add_feature_attribution(
        &mut self,
        idx: u16,
        contribution: FixedPoint,
        description: &[u8; 32],
    ) {
        if self.feature_count as usize >= MAX_FEATURES {
            return;
        }
        self.features[self.feature_count as usize] = FeatureAttribution {
            feature_idx: idx,
            contribution: contribution.abs(),
            sign: if contribution.to_f32() >= 0.0 { 1 } else { -1 },
            description: *description,
        };
        self.feature_count += 1;
    }

    fn update_metrics(&mut self) {
        self.analysis_count += 1;
        if self.edge_count > 0 {
            let mut sum_conf = FixedPoint::ZERO;
            for i in 0..self.edge_count as usize {
                sum_conf += self.edges[i].confidence;
            }
            self.avg_confidence = sum_conf / FixedPoint::from_int(self.edge_count as i32);
        }
        let n =
            crate::core::memory::NEURON_COUNT.load(core::sync::atomic::Ordering::Relaxed) as f32;
        let total_possible = n * n;
        if total_possible > 0.0 {
            self.graph_density = FixedPoint::from_f32((self.edge_count as f32) / total_possible);
        }
    }

    pub fn top_causal_paths(
        &self,
        n: usize,
    ) -> alloc::vec::Vec<(NeuronId, NeuronId, FixedPoint, u32, u8, u8)> {
        let mut result = alloc::vec::Vec::new();
        let mut indices: alloc::vec::Vec<usize> = (0..self.edge_count as usize).collect();
        indices.sort_by(|&a, &b| {
            self.edges[b]
                .confidence
                .partial_cmp(&self.edges[a].confidence)
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        for &i in indices.iter().take(n.min(self.edge_count as usize)) {
            let e = &self.edges[i];
            result.push((
                e.pre,
                e.post,
                e.confidence,
                e.avg_latency,
                e.layer_pair.0,
                e.layer_pair.1,
            ));
        }
        result
    }

    /// Reconstruct a visual ASCII causal path for a given output neuron.
    ///   Returns lines like: "  [L0:004] ──0.92──> [L2:012] ──0.87──> [L6:042]"
    pub fn reconstruct_path_to(
        &self,
        output: NeuronId,
        max_depth: usize,
    ) -> alloc::vec::Vec<alloc::vec::Vec<u8>> {
        let mut lines: alloc::vec::Vec<alloc::vec::Vec<u8>> = alloc::vec::Vec::new();
        let mut depth = 0;
        let mut current = output;
        let mut visited: [bool; 4096] = [false; 4096];
        loop {
            let mut best: Option<(usize, FixedPoint)> = None;
            for i in 0..self.edge_count as usize {
                let e = &self.edges[i];
                if e.post == current && !visited[e.pre.index() as usize] {
                    match best {
                        None => best = Some((i, e.confidence)),
                        Some((_, best_conf)) if e.confidence > best_conf => {
                            best = Some((i, e.confidence))
                        }
                        _ => {}
                    }
                }
            }
            if let Some((best_idx, conf)) = best {
                let e = &self.edges[best_idx];
                visited[current.index() as usize] = true;
                let line = alloc::format!(
                    "  [L{}:{:03}] ──{:.2}──> [L{}:{:03}]",
                    e.layer_pair.0,
                    e.pre.index(),
                    conf.to_f32(),
                    e.layer_pair.1,
                    e.post.index(),
                )
                .into_bytes();
                lines.push(line);
                current = e.pre;
                depth += 1;
                if depth >= max_depth {
                    break;
                }
            } else {
                break;
            }
        }
        lines.reverse();
        lines
    }

    pub fn export_uart_text(&self) -> CausalTextExport {
        let mut buf = [0u8; 2048];
        let mut pos = 0;

        let header = b"HKL1-XAI v1\n";
        for &b in header {
            if pos < 2048 {
                buf[pos] = b;
                pos += 1;
            }
        }

        let meta = format_args!(
            "edges={} density={:.4} conf={:.4}\n",
            self.edge_count,
            self.graph_density.to_f32(),
            self.avg_confidence.to_f32()
        );
        let meta_str = alloc::format!("{}", meta);
        for &b in meta_str.as_bytes() {
            if pos < 2048 {
                buf[pos] = b;
                pos += 1;
            }
        }

        let edge_limit = self.edge_count.min(64);
        for i in 0..edge_limit as usize {
            let e = &self.edges[i];
            let line = alloc::format!(
                "{:04}->{:04} w={:.3} c={:.3} lat={}\n",
                e.pre.index(),
                e.post.index(),
                e.weight.to_f32(),
                e.confidence.to_f32(),
                e.avg_latency
            );
            for &b in line.as_bytes() {
                if pos < 2048 {
                    buf[pos] = b;
                    pos += 1;
                }
            }
        }

        let feat_header = b"\nfeatures:\n";
        for &b in feat_header {
            if pos < 2048 {
                buf[pos] = b;
                pos += 1;
            }
        }

        let feat_limit = self.feature_count.min(32);
        for i in 0..feat_limit as usize {
            let f = &self.features[i];
            let desc = core::str::from_utf8(&f.description).unwrap_or("?");
            let line = alloc::format!(
                "  f{} sign={} cont={:.3} desc={}\n",
                f.feature_idx,
                f.sign,
                f.contribution.to_f32(),
                desc
            );
            for &b in line.as_bytes() {
                if pos < 2048 {
                    buf[pos] = b;
                    pos += 1;
                }
            }
        }

        CausalTextExport {
            data: buf,
            length: pos as u16,
        }
    }

    pub fn export_graphviz_dot(&self) -> CausalTextExport {
        let mut buf = [0u8; 2048];
        let mut pos = 0;

        let header = b"digraph HKL1_Causal {\n  rankdir=LR;\n  node [shape=box style=filled fillcolor=lightyellow];\n";
        for &b in header {
            if pos < 2048 { buf[pos] = b; pos += 1; }
        }

        let edge_limit = self.edge_count.min(128);
        for i in 0..edge_limit as usize {
            let e = &self.edges[i];
            let line = alloc::format!(
                "  n{:04} -> n{:04} [label=\"w={:.2} c={:.2}\" penwidth={:.1}];\n",
                e.pre.index(),
                e.post.index(),
                e.weight.to_f32(),
                e.confidence.to_f32(),
                e.confidence.to_f32().max(0.5),
            );
            for &b in line.as_bytes() {
                if pos < 2048 { buf[pos] = b; pos += 1; }
            }
        }

        let footer = b"}\n";
        for &b in footer {
            if pos < 2048 { buf[pos] = b; pos += 1; }
        }

        CausalTextExport {
            data: buf,
            length: pos as u16,
        }
    }
}

pub struct CausalTextExport {
    pub data: [u8; 2048],
    pub length: u16,
}

impl CausalTextExport {
    pub fn as_str(&self) -> &str {
        core::str::from_utf8(&self.data[..self.length as usize]).unwrap_or("")
    }
}

pub struct SpikeTraceAnalyzer {
    pub causality_pairs: [(u16, u16); 1024],
    pub causality_count: u16,
}

impl SpikeTraceAnalyzer {
    pub fn new() -> Self {
        Self {
            causality_pairs: [(0, 0); 1024],
            causality_count: 0,
        }
    }

    pub fn analyze(
        &mut self,
        trace: &[crate::telemetry::spike_trace::TraceEvent],
        graph: &mut CausalGraph,
    ) {
        for window in trace.windows(3) {
            let a = window[0];
            let b = window[2];
            if b.timestamp > a.timestamp && b.timestamp - a.timestamp < 10 {
                let latency = b.timestamp - a.timestamp;
                graph.update_edge(a.neuron_id, b.neuron_id, latency);
            }
        }
    }

    pub fn analyze_and_record(&mut self, trace: &[crate::telemetry::spike_trace::TraceEvent]) {
        self.causality_count = 0;
        for window in trace.windows(3) {
            let a = window[0];
            let b = window[2];
            if b.timestamp > a.timestamp && b.timestamp - a.timestamp < 10 {
                if self.causality_count < 1024 {
                    let pre_idx = a.neuron_id.index() as u16;
                    let post_idx = b.neuron_id.index() as u16;
                    self.causality_pairs[self.causality_count as usize] = (pre_idx, post_idx);
                    self.causality_count += 1;
                }
            }
        }
    }
}

use core::mem::MaybeUninit;

static mut CAUSAL_GRAPH_STORAGE: MaybeUninit<CausalGraph> = MaybeUninit::uninit();
static mut CAUSAL_ANALYZER_STORAGE: MaybeUninit<SpikeTraceAnalyzer> = MaybeUninit::uninit();

pub fn init_xai() {
    unsafe {
        CAUSAL_GRAPH_STORAGE.write(CausalGraph::new());
        CAUSAL_ANALYZER_STORAGE.write(SpikeTraceAnalyzer::new());
    }
}

pub fn causal_graph() -> &'static mut CausalGraph {
    unsafe { &mut *CAUSAL_GRAPH_STORAGE.as_mut_ptr() }
}

pub fn causal_analyzer() -> &'static mut SpikeTraceAnalyzer {
    unsafe { &mut *CAUSAL_ANALYZER_STORAGE.as_mut_ptr() }
}

pub fn analyze_current_trace() {
    let trace = crate::telemetry::spike_trace::export_trace();
    if !trace.is_empty() {
        let analyzer = causal_analyzer();
        let graph = causal_graph();
        analyzer.analyze(trace, graph);
        analyzer.analyze_and_record(trace);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::memory::NeuronId;

    #[test]
    fn test_causal_graph_new() {
        let g = CausalGraph::new();
        assert_eq!(g.edge_count, 0);
        assert_eq!(g.feature_count, 0);
    }

    #[test]
    fn test_add_edge() {
        let mut g = CausalGraph::new();
        g.add_edge(
            NeuronId::new(1),
            NeuronId::new(2),
            FixedPoint::from_f32(0.5),
            3,
        );
        assert_eq!(g.edge_count, 1);
        assert_eq!(g.edges[0].spike_count, 1);
    }

    #[test]
    fn test_update_edge_increases_count() {
        let mut g = CausalGraph::new();
        g.update_edge(NeuronId::new(1), NeuronId::new(2), 3);
        assert_eq!(g.edge_count, 1);
        g.update_edge(NeuronId::new(1), NeuronId::new(2), 5);
        assert_eq!(g.edges[0].spike_count, 2);
    }

    #[test]
    fn test_add_feature_attribution() {
        let mut g = CausalGraph::new();
        let desc = [0u8; 32];
        g.add_feature_attribution(10, FixedPoint::from_f32(0.8), &desc);
        assert_eq!(g.feature_count, 1);
        assert_eq!(g.features[0].sign, 1);
        assert_eq!(g.features[0].description, desc);
    }

    #[test]
    fn test_export_uart_text() {
        let mut g = CausalGraph::new();
        g.add_edge(
            NeuronId::new(0),
            NeuronId::new(5),
            FixedPoint::from_f32(0.3),
            2,
        );
        let export = g.export_uart_text();
        assert!(export.length > 0);
        let text = export.as_str();
        assert!(text.contains("HKL1-XAI"));
    }

    #[test]
    fn test_export_features() {
        let mut g = CausalGraph::new();
        let mut md = [0u8; 32];
        md[..6].copy_from_slice(b"motion");
        let mut ae = [0u8; 32];
        ae[..4].copy_from_slice(b"edge");
        g.add_feature_attribution(1, FixedPoint::from_f32(0.5), &md);
        g.add_feature_attribution(2, FixedPoint::from_f32(-0.3), &ae);
        let export = g.export_uart_text();
        let text = export.as_str();
        assert!(text.contains("features:"));
        assert!(text.contains("motion"));
        assert!(text.contains("edge"));
    }

    #[test]
    fn test_top_causal_paths() {
        let mut g = CausalGraph::new();
        g.add_edge(NeuronId::new(0), NeuronId::new(1), FixedPoint::ZERO, 1);
        g.add_edge(NeuronId::new(2), NeuronId::new(3), FixedPoint::ZERO, 1);
        let paths = g.top_causal_paths(5);
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn test_multi_edge_density() {
        let mut g = CausalGraph::new();
        crate::core::memory::NEURON_COUNT.store(10, core::sync::atomic::Ordering::Relaxed);
        for i in 0..5 {
            g.add_edge(
                NeuronId::new(i),
                NeuronId::new(i + 1),
                FixedPoint::from_f32(0.1),
                1,
            );
        }
        let density = g.graph_density.to_f32();
        assert!(density > 0.0 && density < 1.0);
        assert_eq!(g.edge_count, 5);
    }

    #[test]
    fn test_update_edge_confidence_grows() {
        let mut g = CausalGraph::new();
        g.update_edge(NeuronId::new(0), NeuronId::new(1), 3);
        let c1 = g.edges[0].confidence.to_f32();
        for _ in 0..100 {
            g.update_edge(NeuronId::new(0), NeuronId::new(1), 3);
        }
        let c2 = g.edges[0].confidence.to_f32();
        assert!(c2 > c1);
    }

    #[test]
    fn test_analyzer_detects_causality() {
        let trace = [
            crate::telemetry::spike_trace::TraceEvent {
                neuron_id: NeuronId::new(0),
                timestamp: 100,
                layer: 1,
                is_predictor: false,
                membrane_potential: 0,
            },
            crate::telemetry::spike_trace::TraceEvent {
                neuron_id: NeuronId::new(1),
                timestamp: 101,
                layer: 1,
                is_predictor: false,
                membrane_potential: 0,
            },
            crate::telemetry::spike_trace::TraceEvent {
                neuron_id: NeuronId::new(2),
                timestamp: 105,
                layer: 2,
                is_predictor: false,
                membrane_potential: 0,
            },
        ];
        let mut graph = CausalGraph::new();
        let mut analyzer = SpikeTraceAnalyzer::new();
        analyzer.analyze(&trace, &mut graph);
        assert_eq!(graph.edge_count, 1);
        assert_eq!(graph.edges[0].pre, NeuronId::new(0));
        assert_eq!(graph.edges[0].post, NeuronId::new(2));
    }

    #[test]
    fn test_analyzer_ignores_distant_spikes() {
        let trace = [
            crate::telemetry::spike_trace::TraceEvent {
                neuron_id: NeuronId::new(0),
                timestamp: 100,
                layer: 1,
                is_predictor: false,
                membrane_potential: 0,
            },
            crate::telemetry::spike_trace::TraceEvent {
                neuron_id: NeuronId::new(1),
                timestamp: 200,
                layer: 1,
                is_predictor: false,
                membrane_potential: 0,
            },
            crate::telemetry::spike_trace::TraceEvent {
                neuron_id: NeuronId::new(2),
                timestamp: 300,
                layer: 2,
                is_predictor: false,
                membrane_potential: 0,
            },
        ];
        let mut graph = CausalGraph::new();
        let mut analyzer = SpikeTraceAnalyzer::new();
        analyzer.analyze(&trace, &mut graph);
        assert_eq!(graph.edge_count, 0);
    }

    #[test]
    fn test_analyze_and_record_stores_pairs() {
        let trace = [
            crate::telemetry::spike_trace::TraceEvent {
                neuron_id: NeuronId::new(0),
                timestamp: 100,
                layer: 1,
                is_predictor: false,
                membrane_potential: 0,
            },
            crate::telemetry::spike_trace::TraceEvent {
                neuron_id: NeuronId::new(1),
                timestamp: 101,
                layer: 1,
                is_predictor: false,
                membrane_potential: 0,
            },
            crate::telemetry::spike_trace::TraceEvent {
                neuron_id: NeuronId::new(2),
                timestamp: 105,
                layer: 2,
                is_predictor: false,
                membrane_potential: 0,
            },
        ];
        let mut analyzer = SpikeTraceAnalyzer::new();
        analyzer.analyze_and_record(&trace);
        assert_eq!(analyzer.causality_count, 1);
        assert_eq!(analyzer.causality_pairs[0], (0, 2));
    }

    #[test]
    fn test_causal_graph_avg_confidence() {
        let mut g = CausalGraph::new();
        g.add_edge(
            NeuronId::new(0),
            NeuronId::new(1),
            FixedPoint::from_f32(0.5),
            2,
        );
        g.add_edge(
            NeuronId::new(2),
            NeuronId::new(3),
            FixedPoint::from_f32(0.5),
            2,
        );
        assert!(g.avg_confidence > FixedPoint::ZERO);
    }

    #[test]
    fn test_init_xai_sets_up_globals() {
        init_xai();
        let g = causal_graph();
        assert_eq!(g.edge_count, 0);
        let a = causal_analyzer();
        assert_eq!(a.causality_count, 0);
    }

    #[test]
    fn test_reconstruct_path_to_basic() {
        let mut g = CausalGraph::new();
        g.add_edge(
            NeuronId::new(0),
            NeuronId::new(2),
            FixedPoint::from_f32(0.5),
            3,
        );
        g.add_edge(
            NeuronId::new(2),
            NeuronId::new(5),
            FixedPoint::from_f32(0.7),
            2,
        );
        let lines = g.reconstruct_path_to(NeuronId::new(5), 10);
        assert!(!lines.is_empty());
        let text = core::str::from_utf8(&lines[0]).unwrap_or("");
        assert!(text.contains("["));
    }

    #[test]
    fn test_reconstruct_path_to_no_path() {
        let g = CausalGraph::new();
        let lines = g.reconstruct_path_to(NeuronId::new(99), 10);
        assert!(lines.is_empty());
    }

    #[test]
    fn test_reconstruct_path_to_cycle_stops() {
        let mut g = CausalGraph::new();
        g.add_edge(NeuronId::new(0), NeuronId::new(1), FixedPoint::ONE, 1);
        g.add_edge(NeuronId::new(1), NeuronId::new(2), FixedPoint::ONE, 1);
        g.add_edge(NeuronId::new(2), NeuronId::new(0), FixedPoint::ONE, 1);
        let lines = g.reconstruct_path_to(NeuronId::new(2), 100);
        assert!(!lines.is_empty());
        assert!(lines.len() <= g.edge_count as usize);
    }

    #[test]
    fn test_export_graphviz_dot() {
        let mut g = CausalGraph::new();
        g.add_edge(NeuronId::new(0), NeuronId::new(1), FixedPoint::from_f32(0.5), 2);
        g.add_edge(NeuronId::new(1), NeuronId::new(2), FixedPoint::from_f32(0.8), 1);
        let dot = g.export_graphviz_dot();
        let text = dot.as_str();
        assert!(text.starts_with("digraph"));
        assert!(text.contains("n0000 -> n0001"));
        assert!(text.contains("n0001 -> n0002"));
        assert!(text.contains("}"));
        assert!(text.len() > 20);
    }
}
