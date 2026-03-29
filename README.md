# Workflow Simulator

A discrete-event simulator for training reinforcement learning agents to manage resource allocation in microservice-based distributed systems. Agents learn to dynamically adjust CPU/thread counts and service placement across nodes to minimize latency violations and resource usage under time-varying load.

## Overview

The simulator models a distributed application as a set of services processing workflow requests (modelled as task graphs). A Q-Learning agent observes system state at configurable intervals and takes actions to scale services up/down or migrate them between nodes. Training runs over many episodes with decaying exploration, and results (Q-tables, metrics) are saved for later analysis.

**Key capabilities:**
- Event-driven simulation with Poisson workflow arrivals and service-to-service call graphs
- Time-varying arrival rates (configurable per workflow, per time window)
- Multi-agent support with per-agent service/node scopes and reward weights
- Configurable action space: CPU scaling, service placement, or both
- Greedy evaluation episodes interleaved with training for unbiased performance measurement

## Project Structure

```
workflow_simulator/
├── agents/
│   ├── agents.py          # QLearningAgent, GlobalAgentDispatcher
│   └── TileCoding.py      # Tile coding feature extraction
├── simulator/
│   ├── application.py     # Service and Workflow definitions
│   ├── infrastructure.py  # Node resource model
│   ├── runtime.py         # Task execution and tracing
│   ├── simulation.py      # Event-driven simulator core
│   └── utils.py           # Utility functions
├── train.py               # Training entry point
├── config.yaml            # Configuration file
└── requirements.txt
```

## Installation

```bash
git clone <repo>
cd workflow_simulator

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

**Dependencies:** `numpy`, `matplotlib`, `networkx`, `PyYAML`, `tqdm`, `scikit-learn`, `scipy`

## Running

```bash
python train.py
# or with a custom config
python train.py --config custom_config.yaml
```

Training saves results to a timestamped directory under `training_results/`:

| File | Contents |
|---|---|
| `qtables.pkl` / `qtables.json` | Learned Q-tables (pickle and human-readable) |
| `metrics.pkl` | Per-episode metrics (rewards, violations, latencies, CPU usage) |
| `metadata.json` | Hyperparameters, agent config, and final metrics |

## Configuration

All behaviour is driven by `config.yaml`. Key sections:

### `topology`

Defines the application as a service call graph:

```yaml
topology:
  n_services: 4
  workflows:
    - id: 0
      lam: {0: 1.0, 30: 1.80}   # Poisson arrival rate, keyed by simulation time (s)
      root_service: 0             # Entry-point service
      edges: [[0, 1], [1, 2]]     # Directed service call graph
```

### `infrastructure`

```yaml
infrastructure:
  n_nodes: 3       # Physical nodes
  cpu_max: 4       # Max CPU cores per node
  ram: 8
  freq: 1000       # CPU frequency
```

### `simulation`

```yaml
simulation:
  iterations: 500       # Events per episode
  timeout: 600000       # Task timeout (ms)
  eval_interval: 10.0   # Seconds between agent evaluations
```

### `rl`

```yaml
rl:
  epsilon: 0.30         # Initial exploration rate
  epsilon_min: 0.05
  epsilon_decay: 0.992  # Per-episode decay
  alpha: 0.10           # Learning rate
  gamma: 0.9            # Discount factor
```

### `training`

```yaml
training:
  n_episodes: 1000
  eval_greedy_every: 10   # Run a greedy (no exploration) episode every N episodes
```

### `control`

Defines which dimensions agents can control and which agent owns each service:

```yaml
control:
  agent_controls_replicas: true    # Enable CPU/thread scaling actions
  agent_controls_placement: false  # Enable service migration actions
  agent_control_assignments:
    s1.scaling: agent1
    s2.scaling: agent1
```

### `reward`

```yaml
reward:
  e2e_lat_target: 8.0              # Latency SLO target (seconds)
  reward_cpu_weight: 0.50
  reward_lat_viol_weight: 0.50
  agents:
    agent1:
      workflows:
        0: 0.30    # Per-workflow latency violation weight
        1: 0.20
      nodes:
        0: 0.20    # Per-node CPU usage weight
        1: 0.15
        2: 0.15
```

## State, Actions, and Reward

**State** (per agent):
- Queue occupancy category per service in scope (empty / low / high)
- Current CPU thread count per service in scope
- Node CPU usage category per node in scope (low / medium / high)

**Actions:** Cartesian product over CPU thread counts `[1, ..., CPU_MAX]` for each controlled service (and optionally node placement choices).

**Reward:**
```
reward = 1.0 - Σ(weighted_objectives)
```
where objectives are per-workflow latency violation rates and per-node CPU usage rates, weighted according to the agent's config.

## Notebooks

| Notebook | Purpose |
|---|---|
| `simulator.ipynb` | Interactive exploration of the simulator |
| `single_eval.ipynb` | Evaluate a single trained policy |
| `comparative_eval.ipynb` | Compare multiple trained policies |
| `plot_results.ipynb` | Plot training curves from saved metrics |
| `training.ipynb` | Run training interactively |
