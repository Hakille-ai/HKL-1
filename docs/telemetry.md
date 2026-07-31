# Telemetry Module

The telemetry module provides observability and interpretability for the HKL-1 system.

## Spike Trace (`telemetry/spike_trace.rs`)

Real-time recording of neural activity.

### Buffer

- `SPIKE_TRACE_BUFFER = 8192` events
- Circular buffer, oldest events overwritten
- Each event: `(timestamp, neuron_id, spike_type)`

### Initialization

`spike_trace::init_logger()` is called at boot (t=3.0ms) to initialize the telemetry buffer before the main loop starts.

### Features

| Feature | Description |
|---|---|
| Per-neuron spike count | Track firing rates |
| Burst detection | Identify burst patterns |
| Population activity | Aggregate firing statistics |
| Trigger modes | Pre/post event capture |
| Filtering | By neuron type, layer, or ID |

### Data Export

Spike trace data can be exported via:
- UART/serial for external analysis (text-format output, 2048 bytes)
- Flash dump for post-mortem analysis
- Real-time streaming to host computer

## XAI — Explainable AI (`telemetry/xai.rs`)

Generates human-readable explanations of network decisions.

### Explanation Types

| Type | Description |
|---|---|
| Feature attribution | Which input neurons contributed to output (128 `FeatureAttribution` slots with contribution + sign) |
| Path discovery | Key synapses in decision pathway (`CausalGraph` with 4096 edges, confidence EMA) |
| Temporal relevance | When input features were most important |
| Neuromodulator state | How mood/arousal affected decision |
| Confidence | Network certainty about output |

### Causal Graph

- `CausalGraph` maintains a directed graph (4096 edges max)
- Edge confidence updated with spike_count + exponential moving average
- `top_causal_paths()` returns sorted paths by confidence (includes confidence, latency, layer pair)
- `reconstruct_path_to(output, max_depth)` traces a causal path backward from output neuron, producing visual ASCII arrows: `[L0:004] ──0.92──> [L2:012] ──0.87──> [L6:042]`
- `export_uart_text()` formats graph as structured text (2048 bytes) for external debugging
- `export_graphviz_dot()` renders the causal graph as Graphviz DOT (128 edges max, 2048 bytes) for visual reconstruction: `n0000 -> n0001 [label="w=0.50 c=0.50" penwidth=0.5]`

### Example Output

```
Decision: AVOID (confidence: 0.87)
  Input drivers:  proximity(0.92), touch_left(0.45)
  Key pathway:   sensor_3 → interneuron_12 → motor_5
  Suppressed by: serotonin (low risk tolerance)
  Temporal peak: 230-250ms after stimulus
```

### Use Cases

- Debugging unexpected behavior
- Safety auditing
- Human-in-the-loop validation
- Post-incident analysis

### Test Coverage

| Module | Tests |
|---|---|
| `telemetry/spike_trace.rs` | 13 |
| `telemetry/xai.rs` | 14 |
