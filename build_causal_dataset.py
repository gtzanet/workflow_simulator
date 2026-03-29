"""
Build a causal discovery dataset from the workflow simulator.

Runs repeated simulations with no agents, collects averaged metrics per run as
CSV rows, computes the NOTEARS causal graph after each run, and stops when the
graph structure converges between consecutive estimates.

Usage:
    python build_causal_dataset.py [--config config.yaml] [--output causal_dataset.csv]
                                   [--iterations N] [--min-samples 10]
                                   [--max-samples 200] [--convergence-threshold 0.05]
"""

import argparse
import sys
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from simulator.simulation import Simulation
from simulator.application import Application
from simulator.infrastructure import Node
from causal_discovery import causal_discovery, get_binary_graph
from plot_causal_graph import plot_causal_graph


# ---------------------------------------------------------------------------
# Passive metrics collector — acts as an "agent" but never changes anything
# ---------------------------------------------------------------------------

class MetricsCollector:
    """Captures accumulated + instant metrics from each eval window."""

    def __init__(self):
        self.windows: list = []

    def on_eval(self, idx, service, accumulated, instant):
        # Called once per service per eval window; we only need one copy per window
        if idx == 0:
            self.windows.append((accumulated, instant))
        return None  # no action taken

    def reset(self):
        self.windows.clear()


# ---------------------------------------------------------------------------
# Row builder — averages metrics across all eval windows in one simulation run
# ---------------------------------------------------------------------------

def _safe_mean(values: list) -> float:
    return float(np.mean(values)) if values else 0.0


def build_row(collector: MetricsCollector, nodes: list, cpu_max: int,
              n_services: int, n_workflows: int) -> dict:
    """Aggregate all eval-window metrics into a single dataset row."""
    row = {}

    for sid in range(n_services):
        # Only average latency/throughput over windows where the service was active
        latencies = [
            w[0]["services"][sid]["avg_latency"]
            for w in collector.windows
            if w[0]["services"][sid]["arrivals"] > 0
        ]
        throughputs = [
            w[0]["services"][sid]["avg_throughput"]
            for w in collector.windows
            if w[0]["services"][sid]["arrivals"] > 0
        ]
        row[f"s{sid}_avg_latency"]     = _safe_mean(latencies)
        row[f"s{sid}_arrival_rate"]    = _safe_mean(
            [w[0]["services"][sid]["arrival_rate"]    for w in collector.windows])
        row[f"s{sid}_departure_rate"]  = _safe_mean(
            [w[0]["services"][sid]["departure_rate"]  for w in collector.windows])
        row[f"s{sid}_avg_throughput"]  = _safe_mean(throughputs)
        row[f"s{sid}_avg_queue_size"]  = _safe_mean(
            [w[1]["services"][sid]["queue_size"]      for w in collector.windows])
        row[f"s{sid}_avg_threads"]     = _safe_mean(
            [w[1]["services"][sid]["threads"]         for w in collector.windows])

    for wid in range(n_workflows):
        # Only include windows where at least one e2e trace completed
        e2e_lats = [
            w[0]["workflows"]["e2e_latencies"][wid]
            for w in collector.windows
            if w[0]["workflows"]["e2e_latencies"][wid] > 0
        ]
        row[f"wf{wid}_avg_e2e_latency"] = _safe_mean(e2e_lats)
        row[f"wf{wid}_violation_rate"]  = _safe_mean(
            [w[0]["workflows"]["violation_rates"][wid] for w in collector.windows])

    for node in nodes:
        cpu_arr = np.array(node.cpu_metric, dtype=float)
        avg_threads = float(np.mean(cpu_arr)) if cpu_arr.size > 0 else 0.0
        row[f"node{node.id}_avg_cpu_threads"] = avg_threads
        row[f"node{node.id}_cpu_usage_pct"]   = min(
            100.0, 100.0 * avg_threads / max(float(cpu_max), 1e-12))

    return row


# ---------------------------------------------------------------------------
# Convergence check
# ---------------------------------------------------------------------------

def graph_converged(W_prev: np.ndarray, W_curr: np.ndarray,
                    threshold: float, binary_threshold: float = 0.1) -> bool:
    """Return True if the fraction of changed edges is <= threshold."""
    B_prev = get_binary_graph(W_prev, binary_threshold)
    B_curr = get_binary_graph(W_curr, binary_threshold)
    changed_fraction = float(np.sum(B_prev != B_curr)) / max(B_prev.size, 1)
    return changed_fraction <= threshold


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build causal dataset from workflow simulator (no agents)"
    )
    parser.add_argument("--config",  default="config.yaml",
                        help="Path to simulation config (default: config.yaml)")
    parser.add_argument("--output",  default="causal_dataset.csv",
                        help="Output CSV path (default: causal_dataset.csv)")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Simulation iterations per run (overrides config)")
    parser.add_argument("--min-samples", type=int, default=10,
                        help="Minimum rows before convergence is checked (default: 10)")
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Hard cap on total simulation runs (default: 200)")
    parser.add_argument("--convergence-threshold", type=float, default=0.05,
                        help="Max fraction of edges that may change to declare "
                             "convergence (default: 0.05)")
    parser.add_argument("--convergence-window", type=int, default=30,
                        help="Number of consecutive stable comparisons required "
                             "to declare convergence (default: 30)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config)")
    args = parser.parse_args()

    # --- Load config (topology + infrastructure + simulation only) ----------
    with open(args.config) as f:
        config = yaml.safe_load(f)

    topology_cfg = config.get("topology", {})
    for wf in topology_cfg.get("workflows", []):
        wf["edges"] = [tuple(e) for e in wf.get("edges", [])]

    infra_cfg  = config.get("infrastructure", {})
    sim_cfg    = config.get("simulation", {})
    reward_cfg = config.get("reward", {})

    N_NODES        = infra_cfg.get("n_nodes", 3)
    CPU_MAX        = infra_cfg.get("cpu_max", 4)
    RAM            = infra_cfg.get("ram", 8)
    FREQ           = infra_cfg.get("freq", 1000)
    ITERATIONS     = args.iterations or sim_cfg.get("iterations", 500)
    TIMEOUT        = sim_cfg.get("timeout", 600000)
    EVAL_INTERVAL  = sim_cfg.get("eval_interval", 10.0)
    LATENCY_TARGET = reward_cfg.get("e2e_lat_target", None)
    SEED           = args.seed if args.seed is not None else sim_cfg.get("seed", 42)

    np.random.seed(SEED)

    # --- Build application and infrastructure --------------------------------
    app = Application(topology=topology_cfg)
    n_services  = len(app.services)
    n_workflows = len(topology_cfg.get("workflows", []))

    nodes = [Node(i, CPU_MAX, RAM, FREQ) for i in range(N_NODES)]
    initial_service_map = {s.id: nodes[s.id % N_NODES] for s in app.services}
    app.deploy_services(initial_service_map)
    for s in app.services:
        s.threads = 1
    app.reset()

    # Register collector as the agent for service 0 only; no other service
    # will have an agent, so the simulator skips them silently.
    collector = MetricsCollector()
    agents = {0: collector}

    # --- Dataset accumulation loop -------------------------------------------
    rows: list  = []
    output_path = Path(args.output)
    W_prev      = None
    col_prev    = None

    print("Building causal dataset")
    print(f"  Config:      {args.config}")
    print(f"  Output:      {output_path}")
    print(f"  Nodes: {N_NODES}  Services: {n_services}  Workflows: {n_workflows}")
    print(f"  Iterations per run: {ITERATIONS}  eval_interval: {EVAL_INTERVAL}s")
    print(f"  Min/max samples: {args.min_samples} / {args.max_samples}")
    print(f"  Convergence threshold: {args.convergence_threshold}  "
          f"(window: {args.convergence_window} consecutive)")
    print()

    consecutive_stable = 0

    for sample_idx in range(args.max_samples):
        # Reset simulation state; app.reset() cascades to service.reset() → node.reset()
        app.reset()
        for s in app.services:
            s.threads = 1
        collector.reset()

        sim = Simulation(
            apps=[app],
            units=[],
            iterations=ITERATIONS,
            timeout=TIMEOUT,
            eval_interval=EVAL_INTERVAL,
            latency_target=LATENCY_TARGET,
        )
        sim.run(agents=agents)

        if not collector.windows:
            print(f"  [{sample_idx + 1:>4}] No eval windows collected — skipping")
            continue

        row = build_row(collector, nodes, CPU_MAX, n_services, n_workflows)
        rows.append(row)

        # Persist CSV after every sample so partial results are never lost
        pd.DataFrame(rows).to_csv(output_path, index=False)

        status = (f"  [{sample_idx + 1:>4}] "
                  f"{len(collector.windows)} windows  →  {len(rows)} rows")

        if len(rows) >= args.min_samples:
            W_curr, col_curr = causal_discovery(str(output_path))

            n_edges = int(np.sum(get_binary_graph(W_curr)))

            if W_prev is not None:
                stable = graph_converged(W_prev, W_curr,
                                         args.convergence_threshold)
                if stable:
                    consecutive_stable += 1
                else:
                    consecutive_stable = 0
                status += (f"  |  edges={n_edges}"
                           f"  stable={consecutive_stable}/{args.convergence_window}")
                print(status)

                if consecutive_stable >= args.convergence_window:
                    W_prev, col_prev = W_curr, col_curr
                    print(f"\nConverged after {len(rows)} samples "
                          f"({args.convergence_window} consecutive stable comparisons).")
                    break
            else:
                status += f"  |  edges={n_edges}  (first graph)"
                print(status)

            W_prev, col_prev = W_curr, col_curr
        else:
            print(status)

    # --- Summary -------------------------------------------------------------
    print(f"\nDataset saved: {output_path}  ({len(rows)} rows)")

    if W_prev is not None:
        print(f"\nFinal causal graph  ({len(col_prev)} variables)")
        print(f"Columns: {col_prev}")
        print("\nWeighted adjacency matrix W  (W[i,j] = i → j):")
        print(np.round(W_prev, 3))
        B = get_binary_graph(W_prev)
        edges = [(col_prev[i], col_prev[j]) for i, j in zip(*np.nonzero(B))]
        print(f"\nBinary graph (threshold=0.1)  —  {len(edges)} edges:")
        for src, dst in edges:
            print(f"  {src} → {dst}")

        # --- Plot ----------------------------------------------------------------
        plot_path = output_path.with_suffix(".png")
        plot_causal_graph(
            W_prev, col_prev,
            title=f"Causal Graph  ({len(rows)} samples)",
            save_path=plot_path,
        )
    else:
        print("Not enough samples to compute a causal graph.")
        sys.exit(1)


if __name__ == "__main__":
    main()
