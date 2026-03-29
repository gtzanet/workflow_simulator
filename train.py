import sys
import os
import json
import pickle
import random
import argparse
import yaml
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tqdm import tqdm

sys.path.insert(0, str(Path.cwd()))

from simulator.application import Application
from simulator.infrastructure import Node
from simulator.simulation import Simulation
from agents.agents import QLearningAgent

parser = argparse.ArgumentParser(description="Run Q-Learning Training for Workflow Simulator")
parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML config file")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# -- Application topology -----------------------------------------------------
APP_TOPOLOGY = config.get("topology", {})
for wf in APP_TOPOLOGY.get("workflows", []):
    wf["edges"] = [tuple(edge) for edge in wf.get("edges", [])]

N_SERVICES  = APP_TOPOLOGY.get("n_services", 4)
N_WORKFLOWS = len(APP_TOPOLOGY.get("workflows", []))

# -- Infrastructure ------------------------------------------------------------
infra_config = config.get("infrastructure", {})
N_NODES = infra_config.get("n_nodes", 3)
CPU_MAX = infra_config.get("cpu_max", 4)
RAM     = infra_config.get("ram", 8)
FREQ    = infra_config.get("freq", 1000)

# -- Simulation run ------------------------------------------------------------
sim_config = config.get("simulation", {})
ITERATIONS    = sim_config.get("iterations", 600)
TIMEOUT       = sim_config.get("timeout", 600000)
EVAL_INTERVAL = sim_config.get("eval_interval", 10.0)
SEED          = sim_config.get("seed", 42)

# -- RL hyperparameters --------------------------------------------------------
rl_config = config.get("rl", {})
EPSILON       = rl_config.get("epsilon", 0.30)
EPSILON_MIN   = rl_config.get("epsilon_min", 0.05)
EPSILON_DECAY = rl_config.get("epsilon_decay", 0.992)
ALPHA         = rl_config.get("alpha", 0.10)
GAMMA         = rl_config.get("gamma", 0.9)

# -- Training ------------------------------------------------------------------
train_config = config.get("training", {})
N_EPISODES        = train_config.get("n_episodes", 1000)
EVAL_GREEDY_EVERY = train_config.get("eval_greedy_every", 10)

# -- Control flags -------------------------------------------------------------
ctrl_config = config.get("control", {})
AGENT_CONTROLS_REPLICAS  = ctrl_config.get("agent_controls_replicas", True)
AGENT_CONTROLS_PLACEMENT = ctrl_config.get("agent_controls_placement", False)
AGENT_CONTROL_ASSIGNMENTS = ctrl_config.get("agent_control_assignments", {})

# -- Reward normalisation and weighting ----------------------------------------
reward_config = config.get("reward", {})
E2E_LAT_TARGET = reward_config.get("e2e_lat_target", 4.0)
REWARD_CPU_WEIGHT = reward_config.get("reward_cpu_weight", 0.50)
REWARD_LAT_VIOL_WEIGHT = reward_config.get("reward_lat_viol_weight", 0.50)
REWARD_AGENT_CONFIG = reward_config.get("agents", {})  # agent_name -> {workflows: {wid: w}, nodes: {nid: w}}

RESULTS_DIR = Path("training_results")

np.random.seed(SEED)
random.seed(SEED)
print("Configuration OK")
print(f"  n_services={N_SERVICES}  n_workflows={N_WORKFLOWS}  n_nodes={N_NODES}  cpu_max={CPU_MAX}")
print(f"  active strategy: {sorted(set(AGENT_CONTROL_ASSIGNMENTS.values()))}")
print(f"  reward weights: cpu={REWARD_CPU_WEIGHT} latency_violation={REWARD_LAT_VIOL_WEIGHT} per_agent_overrides={sorted(REWARD_AGENT_CONFIG.keys())}")


app = Application(topology=APP_TOPOLOGY)
print(f"Services : {len(app.services)}")
print(f"Workflows: {len(app.workflows)}")

# Infrastructure nodes
nodes = [Node(i, CPU_MAX, RAM, FREQ) for i in range(N_NODES)]

# Initial service → node mapping (round-robin)
initial_service_map = {s.id: nodes[s.id % N_NODES] for s in app.services}
app.deploy_services(initial_service_map)

# Give each service the full thread budget at the start
for s in app.services:
    s.threads = CPU_MAX

app.reset()

print()
print("Initial placement:")
for s in app.services:
    print(f"  service {s.id} → node {s.node.id}  threads={s.threads}")



from itertools import product


def normalize_control_name(name):
    n = str(name).strip().lower()
    if n in ("scaling", "replicas", "cpu"):
        return "cpu"
    if n in ("migration", "placement", "node"):
        return "node"
    raise ValueError(f"Unsupported control name: {name}")


def parse_service_token(token, service_ids):
    t = str(token).strip().lower()
    if not t.startswith("s") or len(t) < 2:
        raise ValueError(f"Invalid service token: {token}")

    raw_num = int(t[1:])

    # If service id 0 exists, default to 1-based notation (s1 -> 0).
    if 0 in service_ids and (raw_num - 1) in service_ids:
        return raw_num - 1

    if raw_num in service_ids:
        return raw_num
    if (raw_num - 1) in service_ids:
        return raw_num - 1

    raise ValueError(
        f"Service token '{token}' does not match service ids {service_ids}. "
        f"Accepted forms are s<id> or 1-based s<id+1>."
    )


def normalize_assignments(raw_assignments, service_ids):
    by_agent = {}

    for key, agent_name in raw_assignments.items():
        parts = str(key).split(".", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid assignment key '{key}'. Expected format 'sX.control'.")

        service_token, control_token = parts
        sid = parse_service_token(service_token, service_ids)
        control = normalize_control_name(control_token)

        agent_key = str(agent_name).strip()
        if not agent_key:
            raise ValueError(f"Invalid empty agent name for key '{key}'.")

        by_agent.setdefault(agent_key, set()).add((sid, control))

    return by_agent


def validate_assignments(by_agent, service_ids, active_service_ids):
    enabled_controls = set()
    if AGENT_CONTROLS_REPLICAS:
        enabled_controls.add("cpu")
    if AGENT_CONTROLS_PLACEMENT:
        enabled_controls.add("node")

    required_pairs = {(sid, ctrl) for sid in active_service_ids for ctrl in enabled_controls}

    assigned_pairs = set()
    duplicates = set()
    invalid_controls = set()
    inactive_assignments = set()

    for pairs in by_agent.values():
        for sid, ctrl in pairs:
            if ctrl not in enabled_controls:
                invalid_controls.add((sid, ctrl))
            if sid not in active_service_ids:
                inactive_assignments.add((sid, ctrl))
            if (sid, ctrl) in assigned_pairs:
                duplicates.add((sid, ctrl))
            assigned_pairs.add((sid, ctrl))

    if invalid_controls:
        raise ValueError(
            "Assignments include disabled controls: "
            + ", ".join(f"s{sid}.{ctrl}" for sid, ctrl in sorted(invalid_controls))
        )

    if inactive_assignments:
        raise ValueError(
            "Assignments include inactive services (no workflow traffic): "
            + ", ".join(f"s{sid}.{ctrl}" for sid, ctrl in sorted(inactive_assignments))
        )

    missing = sorted(required_pairs - assigned_pairs)
    if missing:
        raise ValueError(
            "Missing active service/control assignments: "
            + ", ".join(f"s{sid}.{ctrl}" for sid, ctrl in missing)
        )

    if duplicates:
        raise ValueError(
            "Duplicate service/control assignments detected: "
            + ", ".join(f"s{sid}.{ctrl}" for sid, ctrl in sorted(duplicates))
        )


def build_agent_action_space(responsibilities, node_pool, cpu_actions):
    # responsibilities: [(service_id, 'cpu'|'node'), ...]
    dimensions = []
    for sid, control in responsibilities:
        options = list(cpu_actions) if control == "cpu" else list(node_pool)
        dimensions.append((sid, control, options))

    if not dimensions:
        return [{}]

    option_lists = [opts for _, _, opts in dimensions]
    action_space = []

    for combo in product(*option_lists):
        action = {}
        for (sid, control, _), value in zip(dimensions, combo):
            action.setdefault(sid, {})[control] = value
        action_space.append(action)

    return action_space


service_ids = [s.id for s in app.services]
active_service_ids = sorted({int(app.task_graph.nodes[n]["subset"]) for n in app.task_graph.nodes})

cpu_action_set = list(range(1, CPU_MAX + 1))
node_pool_all = [node.id for node in nodes]

# Build service → workflows mapping: which workflow IDs involve each service.
service_to_workflows = {}
for wf in app.workflows:
    for task_node in wf.nodes:
        sid = int(app.task_graph.nodes[task_node]["subset"])
        service_to_workflows.setdefault(sid, set()).add(wf.id)
service_to_workflows = {k: sorted(v) for k, v in service_to_workflows.items()}

assignments_by_agent = normalize_assignments(AGENT_CONTROL_ASSIGNMENTS, service_ids)
validate_assignments(assignments_by_agent, service_ids, active_service_ids)

agent_specs = []
for agent_name in sorted(assignments_by_agent.keys()):
    responsibilities = sorted(assignments_by_agent[agent_name], key=lambda x: (x[0], x[1]))
    action_space = build_agent_action_space(responsibilities, node_pool_all, cpu_action_set)

    # Collect the workflow IDs that pass through any of this agent's services.
    s_scope = {sid for sid, _ in responsibilities}
    workflow_scope = sorted({wid for sid in s_scope for wid in service_to_workflows.get(sid, [])})

    node_scope = sorted({initial_service_map[sid].id for sid in s_scope})

    agent_specs.append({
        "name": agent_name,
        "responsibilities": responsibilities,
        "action_space": action_space,
        "workflow_scope": workflow_scope,
        "node_scope": node_scope,
    })

print("Active services in workflows:", active_service_ids)
print("Service → workflows map:", service_to_workflows)
print("Global agent/action setup:")
for i, spec in enumerate(agent_specs):
    readable_resp = [f"s{sid}.{ctrl}" for sid, ctrl in spec["responsibilities"]]
    print(
        f"  global_agent[{i}] name={spec['name']} "
        f"responsibilities={readable_resp} n_actions={len(spec['action_space'])} "
        f"workflow_scope={spec['workflow_scope']} node_scope={spec['node_scope']}"
    )
    print(f"    sample action: {spec['action_space'][0]}")



def make_agent_get_state(responsibilities, node_scope, cpu_cap=20):
    """
    State layout: (queue_cat per service..., cpu_threads per service..., node_cpu_cat per node...)
      - queue_cat    : 0=empty, 1=low (1-4), 2=high (>=5)
      - cpu_threads  : int threads, capped at cpu_cap (default 20)
      - node_cpu_cat : 0=low (<33%), 1=medium (33-66%), 2=high (>66%)

    Length = 2 * len(service_scope) + len(node_scope) — fixed at creation time.
    """
    service_scope = sorted({sid for sid, _ in responsibilities})
    node_scope = sorted(node_scope)

    def _get_state(self, observations):
        queues = observations.get("service_queues", [])
        cpus = observations.get("service_cpus", [])
        node_cpu_usage_pct = observations.get("node_cpu_usage_pct", {})

        state = []

        # Queue category per service
        for sid in service_scope:
            q_val = int(queues[sid])
            if q_val == 0:
                state.append(0)
            elif q_val <= 4:
                state.append(1)
            else:
                state.append(2)

        # CPU thread count per service
        for sid in service_scope:
            state.append(min(int(cpus[sid]), cpu_cap))

        # Node CPU usage category per node
        for nid in node_scope:
            usage = float(node_cpu_usage_pct.get(nid, 0.0))
            if usage < 0.33:
                state.append(0)
            elif usage <= 0.66:
                state.append(1)
            else:
                state.append(2)

        return tuple(state)

    return _get_state


def make_agent_reward(workflow_scope, node_scope, agent_reward_cfg=None):
    """
    Reward = 1 - Σ(w_i * objective_i), where weights are normalised to sum to 1.

    Objectives and their weights (resolved in priority order):
      1. Per-agent config  (agent_reward_cfg): {workflows: {wid: w}, nodes: {nid: w}}
      2. Global fallback   (REWARD_CPU_WEIGHT / REWARD_LAT_VIOL_WEIGHT):
           equal weight per workflow scaled by lat_w, equal weight per node scaled by cpu_w.

    node_scope is fixed at agent creation time (from initial_service_map).
    """
    node_scope = sorted(node_scope)
    lat_target = max(float(E2E_LAT_TARGET), 1e-12)

    if agent_reward_cfg is not None:
        # Per-entity weights from config; YAML parses integer keys as ints.
        raw_wf_weights  = {int(k): float(v) for k, v in agent_reward_cfg.get("workflows", {}).items()}
        raw_node_weights = {int(k): float(v) for k, v in agent_reward_cfg.get("nodes", {}).items()}
        # Fall back to 1.0 for any entity not listed.
        wf_weights   = {wid: raw_wf_weights.get(wid, 1.0)  for wid in workflow_scope}
        node_weights = {nid: raw_node_weights.get(nid, 1.0) for nid in node_scope}
    else:
        # Global fallback: equal weight per workflow / per node, scaled by global weights.
        n_wf   = max(len(workflow_scope), 1)
        n_node = max(len(node_scope), 1)
        total_w = float(REWARD_CPU_WEIGHT) + float(REWARD_LAT_VIOL_WEIGHT)
        if total_w <= 0:
            cpu_w, lat_w = 0.5, 0.5
        else:
            cpu_w = float(REWARD_CPU_WEIGHT) / total_w
            lat_w = float(REWARD_LAT_VIOL_WEIGHT) / total_w
        wf_weights   = {wid: lat_w / n_wf   for wid in workflow_scope}
        node_weights = {nid: cpu_w / n_node  for nid in node_scope}

    # Normalise all weights together so they sum to 1.
    total_w = sum(wf_weights.values()) + sum(node_weights.values())
    if total_w <= 0:
        total_w = 1.0
    wf_weights   = {k: v / total_w for k, v in wf_weights.items()}
    node_weights = {k: v / total_w for k, v in node_weights.items()}

    def _reward(self, observations):
        workflow_violation_rates = observations.get("workflow_violation_rates", [])
        workflow_e2e_means = observations.get("workflow_e2e_means", [])
        node_cpu_usage_pct = observations.get("node_cpu_usage_pct", {})

        penalty = 0.0

        # Workflow latency violation rate (per workflow)
        for wid in workflow_scope:
            if wid < len(workflow_violation_rates):
                viol_rate = float(workflow_violation_rates[wid])
            else:
                e2e = float(workflow_e2e_means[wid]) if wid < len(workflow_e2e_means) else 0.0
                viol_rate = max(0.0, (e2e - lat_target) / lat_target)
            penalty += wf_weights[wid] * min(1.0, max(0.0, viol_rate))

        # Node CPU usage rate (per node, fixed scope)
        for nid in node_scope:
            usage = min(1.0, max(0.0, float(node_cpu_usage_pct.get(nid, 0.0))))
            penalty += node_weights[nid] * usage

        return 1.0 - float(penalty)

    return _reward


global_agents = []
for spec in agent_specs:
    n_actions = len(spec["action_space"])
    agent = QLearningAgent(
        epsilon=EPSILON,
        alpha=ALPHA,
        gamma=GAMMA,
        actions=list(range(n_actions)),
    )

    agent_reward_cfg = REWARD_AGENT_CONFIG.get(spec["name"], None)

    agent.get_state = make_agent_get_state(
        spec["responsibilities"],
        spec["node_scope"],
    ).__get__(agent, QLearningAgent)
    agent.reward = make_agent_reward(
        spec["workflow_scope"],
        spec["node_scope"],
        agent_reward_cfg,
    ).__get__(agent, QLearningAgent)

    global_agents.append({
        "name": spec["name"],
        "responsibilities": spec["responsibilities"],
        "workflow_scope": spec["workflow_scope"],
        "node_scope": spec["node_scope"],
        "action_space": spec["action_space"],
        "agent": agent,
    })

for i, item in enumerate(global_agents):
    readable_resp = [f"s{sid}.{ctrl}" for sid, ctrl in item["responsibilities"]]
    print(
        f"global_agent[{i}] name={item['name']} responsibilities={readable_resp} "
        f"workflow_scope={item['workflow_scope']} node_scope={item['node_scope']} "
        f"n_actions={len(item['action_space'])} epsilon={EPSILON} alpha={ALPHA} gamma={GAMMA}"
    )



class GlobalAgentDispatcher:
    """
    Single dispatcher shared by all services.
    The simulator still calls on_eval(service_idx, ...), but this dispatcher
    computes a global action plan once per eval cycle and returns the part for
    each requested service.
    """

    def __init__(self, service_ids, global_agents):
        self.service_ids = list(service_ids)
        self.global_agents = global_agents
        self._eval_counter = 0
        self._planned_actions = {}
        self._last_node_metric_idx = {node.id: 0 for node in nodes}
        self.episode_queue_history = {sid: [] for sid in self.service_ids}

    def _build_observations(self, accumulated_metrics, instant_metrics):
        services_now = instant_metrics["services"]
        services_acc = accumulated_metrics["services"]

        queues = [s["queue_size"] for s in services_now]
        cpus = [s["cpu"] for s in services_now]
        node_ids = [s["node_id"] for s in services_now]

        service_latency_means = [float(s["avg_latency"]) for s in services_acc]
        service_throughput_means = [float(s["avg_throughput"]) for s in services_acc]

        avg_latency_mean = sum(service_latency_means) / len(service_latency_means)
        avg_throughput_mean = sum(service_throughput_means) / len(service_throughput_means)

        node_queue_totals = {}
        for sid, nid in enumerate(node_ids):
            node_queue_totals[nid] = node_queue_totals.get(nid, 0.0) + float(queues[sid])

        for sid in self.service_ids:
            self.episode_queue_history[sid].append(float(queues[sid]))

        # Use actual node CPU usage over the latest eval window.
        node_cpu_usage_pct = {}
        node_avg_threads = {}
        for node in nodes:
            arr = np.array(node.cpu_metric, dtype=float)
            start = int(self._last_node_metric_idx.get(node.id, 0))
            seg = arr[start:] if start < arr.size else np.array([], dtype=float)
            avg_cpu = float(np.mean(seg)) if seg.size > 0 else 0.0
            node_avg_threads[node.id] = avg_cpu
            node_cpu_usage_pct[node.id] = min(1.0, max(0.0, avg_cpu / max(float(CPU_MAX), 1e-12)))
            self._last_node_metric_idx[node.id] = arr.size

        workflow_e2e_means = accumulated_metrics["workflows"].get("e2e_latencies", [])
        workflow_violation_rates = accumulated_metrics["workflows"].get("violation_rates", [])

        return {
            "service_queues": queues,
            "service_cpus": cpus,
            "service_node_ids": node_ids,
            "service_latency_means": service_latency_means,
            "service_throughput_means": service_throughput_means,
            "node_queue_totals": node_queue_totals,
            "node_cpu_usage_pct": node_cpu_usage_pct,
            "node_avg_threads": node_avg_threads,
            "workflow_e2e_means": workflow_e2e_means,
            "workflow_violation_rates": workflow_violation_rates,
            "avg_queue": sum(queues) / len(queues),
            "avg_latency_mean": avg_latency_mean,
            "avg_throughput_mean": avg_throughput_mean,
            "avg_success_rate": accumulated_metrics["workflows"]["avg_success_rate"],
        }

    def _compute_plan(self, accumulated_metrics, instant_metrics):
        observations = self._build_observations(accumulated_metrics, instant_metrics)
        combined = {sid: {} for sid in self.service_ids}

        for item in self.global_agents:
            agent = item["agent"]
            action_space = item["action_space"]

            action_idx = int(agent.step(observations))
            action_idx = max(0, min(action_idx, len(action_space) - 1))
            partial_plan = action_space[action_idx]

            for sid, partial_action in partial_plan.items():
                combined[sid].update(partial_action)

        self._planned_actions = combined

    def on_eval(self, service_idx, service, accumulated_metrics, instant_metrics):
        if self._eval_counter % len(self.service_ids) == 0:
            self._compute_plan(accumulated_metrics, instant_metrics)
        self._eval_counter += 1

        action = self._planned_actions.get(service.id, {})
        return action if action else None

    def reset_environment(self):
        for item in self.global_agents:
            item["agent"].reset_environment()
        self._last_node_metric_idx = {node.id: 0 for node in nodes}
        self._eval_counter = 0

    def get_epsilons(self):
        return [item["agent"].epsilon for item in self.global_agents]

    def set_epsilons(self, epsilons, reset=False):
        for item, eps in zip(self.global_agents, epsilons):
            item["agent"].epsilon = eps
            if reset:
                item["agent"].reset_environment()

    def total_reward(self):
        return sum(sum(item["agent"].rewards) for item in self.global_agents)

    def qtables(self):
        return {item["name"]: item["agent"].q_table for item in self.global_agents}


# One global dispatcher instance, wired into simulator's existing interface.
global_dispatcher = GlobalAgentDispatcher(service_ids, global_agents)
dispatchers = {sid: global_dispatcher for sid in service_ids}

print("Global dispatcher registered for services:", list(dispatchers.keys()))
print("Internal global agents:", len(global_dispatcher.global_agents))


episode_total_rewards      = []
episode_per_agent_rewards  = {item["name"]: [] for item in global_agents}
greedy_success_rates       = []
greedy_episodes            = []

# Additional episode-level metrics for plotting
episode_workflow_violation_pct = {wf.id: [] for wf in app.workflows}
episode_workflow_avg_latency = {wf.id: [] for wf in app.workflows}
episode_avg_violation_pct = []
episode_node_cpu_usage_pct = {node.id: [] for node in nodes}
episode_node_avg_threads = {node.id: [] for node in nodes}
episode_service_queues = {s.id: [] for s in app.services}

# Build a task_id -> workflow_id lookup once (task ids are unique across workflows).
task_to_workflow = {}
for wf in app.workflows:
    for task_id in wf.nodes:
        task_to_workflow[int(task_id)] = int(wf.id)

# Root task per workflow for e2e latency extraction from sim.history.
workflow_root_task = {int(wf.id): int(wf.initial_task.id) for wf in app.workflows}

pbar = tqdm(range(N_EPISODES), desc="Train", unit="ep", file=sys.stdout)
for ep in pbar:
    linear_eps = max(0.0, 1.0 - (ep / max(1, N_EPISODES - 1)))
    global_dispatcher.set_epsilons([linear_eps] * len(global_dispatcher.get_epsilons()), reset=False)

    # -- 1. Reset environment to its initial state -----------------------------
    for s in app.services:
        s.node = initial_service_map[s.id]
        s.threads = CPU_MAX
    app.reset()

    # -- 2. Reset agent episode state (Q-table is preserved) -------------------
    global_dispatcher.reset_environment()

    # -- 3. Run one training episode -------------------------------------------
    sim = Simulation(
        [app], [],
        ITERATIONS,
        step_size=ITERATIONS,
        alloc_step_size=1,
        timeout=TIMEOUT,
        eval_interval=EVAL_INTERVAL,
        latency_target=E2E_LAT_TARGET,
    )
    sim.run(agents=dispatchers)

    # -- 4. Collect per-agent and total rewards --------------------------------
    ep_total = global_dispatcher.total_reward()
    episode_total_rewards.append(ep_total)

    for item in global_agents:
        episode_per_agent_rewards[item["name"]].append(sum(item["agent"].rewards))

    # -- 4b. Collect per-workflow latency and violations from individual requests
    per_wf_lat_samples = {wf.id: [] for wf in app.workflows}
    per_wf_viol_samples = {wf.id: [] for wf in app.workflows}
    total_req = 0
    total_viol = 0

    for trace_data in sim.history.values():
        if int(trace_data.get("status", -1)) != 1:
            continue

        # Find the workflow this trace belongs to from any task id key.
        task_keys = [k for k in trace_data.keys() if k != "status"]
        if not task_keys:
            continue
        wf_id = None
        for tkey in task_keys:
            wf_id = task_to_workflow.get(int(tkey))
            if wf_id is not None:
                break
        if wf_id is None:
            continue

        root_task_id = workflow_root_task.get(wf_id)
        root_times = trace_data.get(root_task_id)
        if root_times is None or len(root_times) < 2:
            continue

        e2e_latency = float(root_times[-1] - root_times[0])
        violated = 1.0 if e2e_latency > E2E_LAT_TARGET else 0.0

        per_wf_lat_samples[wf_id].append(e2e_latency)
        per_wf_viol_samples[wf_id].append(violated)
        total_req += 1
        total_viol += int(violated)

    for wf in app.workflows:
        lat_samples = per_wf_lat_samples[wf.id]
        viol_samples = per_wf_viol_samples[wf.id]
        episode_workflow_avg_latency[wf.id].append(float(np.mean(lat_samples)) if lat_samples else 0.0)
        episode_workflow_violation_pct[wf.id].append(100.0 * float(np.mean(viol_samples)) if viol_samples else 0.0)

    avg_viol_pct = (100.0 * total_viol / total_req) if total_req > 0 else 0.0
    episode_avg_violation_pct.append(avg_viol_pct)

    # -- 4c. Collect per-node CPU usage and avg threads -------------------------
    for node in nodes:
        cpu_arr = np.array(node.cpu_metric, dtype=float)
        avg_threads = float(np.mean(cpu_arr)) if cpu_arr.size > 0 else 0.0
        usage_pct = 100.0 * avg_threads / max(float(CPU_MAX), 1e-12)
        episode_node_avg_threads[node.id].append(avg_threads)
        episode_node_cpu_usage_pct[node.id].append(min(100.0, max(0.0, usage_pct)))

    for sid in range(N_SERVICES):
        hist = global_dispatcher.episode_queue_history.get(sid, [])
        avg_q = float(np.mean(hist)) if hist else 0.0
        episode_service_queues[sid].append(avg_q)
        global_dispatcher.episode_queue_history[sid] = []

    # -- 5. Epsilon decay for continued exploration -> exploitation ------------
    # (Removed exponential decay, replaced with linear decay at start of ep)

    # -- 6. Periodic greedy evaluation (no exploration) ------------------------
    if (ep + 1) % EVAL_GREEDY_EVERY == 0:
        saved_eps = global_dispatcher.get_epsilons()
        global_dispatcher.set_epsilons([0.0] * len(saved_eps), reset=True)

        for s in app.services:
            s.node = initial_service_map[s.id]
            s.threads = CPU_MAX
        app.reset()

        eval_sim = Simulation(
            [app], [],
            ITERATIONS,
            step_size=ITERATIONS,
            alloc_step_size=1,
            timeout=TIMEOUT,
            eval_interval=EVAL_INTERVAL,
            latency_target=E2E_LAT_TARGET,
        )
        eval_sim.run(agents=dispatchers)

        completed = sum(1 for v in eval_sim.history.values() if v.get("status") == 1)
        total_tr = len(eval_sim.history)
        success = 100.0 * completed / total_tr if total_tr > 0 else 0.0
        greedy_success_rates.append(success)
        greedy_episodes.append(ep + 1)

        global_dispatcher.set_epsilons(saved_eps, reset=False)

        agent_strs = [f"{name}={rw_list[-1]:.1f}" for name, rw_list in episode_per_agent_rewards.items()]
        pbar.set_postfix_str(
            f"{' '.join(agent_strs)} "
            f"v={avg_viol_pct:.0f}% "
            f"g={success:.0f}% "
            f"eps={saved_eps[0]:.2f}"
        )

print("\nTraining complete.")



run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = RESULTS_DIR / f"run_{run_ts}"
run_dir.mkdir(parents=True, exist_ok=True)

shutil.copy(args.config, run_dir / "config.yaml")

# -- Q-tables: pickle (canonical) ---------------------------------------------
qtables = global_dispatcher.qtables()

qtables_pkl_path = run_dir / "qtables.pkl"
with open(qtables_pkl_path, "wb") as f:
    pickle.dump(qtables, f)

# -- Q-tables: JSON (human-readable) ------------------------------------------
def q_table_to_json(q_table):
    return {
        str(k): {str(a): float(v) for a, v in actions.items()}
        for k, actions in q_table.items()
    }

qtables_json_path = run_dir / "qtables.json"
with open(qtables_json_path, "w") as f:
    json.dump(
        {name: q_table_to_json(qt) for name, qt in qtables.items()},
        f,
        indent=2,
    )

# -- Responsibilities: one entry per global agent -----------------------------
responsibilities = []
for item in global_agents:
    responsibilities.append(
        {
            "agent_name": item["name"],
            "assignments": [
                {"service_id": sid, "control": ctrl}
                for sid, ctrl in item["responsibilities"]
            ],
            "workflow_scope": item["workflow_scope"],
            "node_scope": item["node_scope"],
            "n_actions": len(item["action_space"]),
            "sample_action": item["action_space"][0] if item["action_space"] else {},
            "q_table_states_visited": len(item["agent"].q_table),
        }
    )

workflow_info = []
for wf in app.workflows:
    workflow_info.append(
        {
            "workflow_id": wf.id,
            "task_nodes": wf.nodes,
            "arrival_rate_lambda": wf.lam,
        }
    )

metadata = {
    "created_at": run_ts,
    "training": {
        "n_episodes": N_EPISODES,
        "iterations_per_episode": ITERATIONS,
        "eval_interval": EVAL_INTERVAL,
        "seed": SEED,
        "final_episode_reward": float(episode_total_rewards[-1]),
        "best_episode_reward": float(max(episode_total_rewards)),
        "final_greedy_success_rate": float(greedy_success_rates[-1]) if greedy_success_rates else None,
        "best_greedy_success_rate": float(max(greedy_success_rates)) if greedy_success_rates else None,
    },
    "agent_config": {
        "type": "QLearningAgent",
        "dispatcher": "GlobalAgentDispatcher",
        "epsilon_init": EPSILON,
        "epsilon_min": EPSILON_MIN,
        "epsilon_decay": EPSILON_DECAY,
        "alpha": ALPHA,
        "gamma": GAMMA,
        "e2e_latency_target": E2E_LAT_TARGET,
        "reward_cpu_weight": REWARD_CPU_WEIGHT,
        "reward_latency_violation_weight": REWARD_LAT_VIOL_WEIGHT,
        "state_shape": "agent-scoped [service_queues..., service_cpus..., node_cpu_cats...]",
        "reward_shape": (
            "1 - Σ(w_i * objective_i): per-workflow latency violation rates "
            "and per-node CPU usage rates, weights normalised to sum to 1"
        ),
        "reward_agent_config": REWARD_AGENT_CONFIG,
        "assignment_input": AGENT_CONTROL_ASSIGNMENTS,
    },
    "simulation_config": {
        "n_services": len(app.services),
        "n_workflows": len(app.workflows),
        "n_nodes": N_NODES,
        "cpu_max": CPU_MAX,
        "ram": RAM,
        "freq": FREQ,
        "timeout": TIMEOUT,
    },
    "responsibilities": responsibilities,
    "workflows": workflow_info,
    "files": {
        "qtables_pickle": "qtables.pkl",
        "qtables_json": "qtables.json",
        "metadata": "metadata.json",
    },
}

meta_path = run_dir / "metadata.json"
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Model saved -> {run_dir}/")
print(f"  qtables.pkl    {qtables_pkl_path.stat().st_size:>8,} bytes")
print(f"  qtables.json   {qtables_json_path.stat().st_size:>8,} bytes")
print(f"  metadata.json  {meta_path.stat().st_size:>8,} bytes")

# -- Save Plotting Metrics -------------------------------------------------------------
metrics_data = {
    "episode_total_rewards": episode_total_rewards,
    "episode_per_agent_rewards": episode_per_agent_rewards,
    "greedy_success_rates": greedy_success_rates,
    "greedy_episodes": greedy_episodes,
    "episode_workflow_violation_pct": episode_workflow_violation_pct,
    "episode_workflow_avg_latency": episode_workflow_avg_latency,
    "episode_avg_violation_pct": episode_avg_violation_pct,
    "episode_node_cpu_usage_pct": episode_node_cpu_usage_pct,
    "episode_node_avg_threads": episode_node_avg_threads,
    "episode_service_queues": episode_service_queues,
    "N_EPISODES": N_EPISODES,
}
metrics_path = run_dir / "metrics.pkl"
with open(metrics_path, "wb") as f:
    pickle.dump(metrics_data, f)
print(f"  metrics.pkl    {metrics_path.stat().st_size:>8,} bytes")
