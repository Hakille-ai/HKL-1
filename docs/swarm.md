# Swarm Module

The swarm module enables multi-device coordination and collective intelligence.

## Federated Learning (`swarm/federated.rs`)

Distributed learning across a swarm of HKL-1 devices.

### Process

1. Each device trains locally (online learning)
2. Periodically, devices share weight snapshots
3. Coordinator averages weights with differential privacy
4. Devices update local weights from averaged model

### Differential Privacy

- Gaussian noise injected before sharing
- Noise scale configurable per federation round
- Prevents reconstruction of individual training data

### Key Components

| Component | Description |
|---|---|
| `FederatedLearning` | FL coordinator with noise injection |
| `add_dp_noise(weights)` | Add Gaussian noise for privacy |
| `federated_average(node_weights, num_nodes)` | Weighted averaging of peer updates |

## Mesh Networking (`swarm/mesh.rs`)

Decentralized peer-to-peer communication.

### Features

- Peer discovery (beacon/ping)
- Gossip protocol for state propagation
- Topology management (mesh formation)
- Message routing (flooding or directed)

### Message Types

| Type | Purpose |
|---|---|
| Beacon | Presence announcement |
| WeightShare | Synaptic weight snapshot |
| SpikeEvent | Remote spike propagation |
| Command | Remote configuration |
| Sync | Clock/time synchronization |
