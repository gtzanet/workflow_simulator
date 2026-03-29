"""
Causal graph plotting utilities.


Provides human-readable node labels, colour-coded node groups, and a
publication-quality layout for the NOTEARS adjacency matrices produced by
causal_discovery.py / build_causal_dataset.py.
"""
from __future__ import annotations

import re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from causal_discovery import get_binary_graph


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

_METRIC_NAMES = {
    "avg_latency":    "Avg Latency",
    "arrival_rate":   "Arrival Rate",
    "departure_rate": "Departure Rate",
    "avg_throughput": "Throughput",
    "avg_queue_size": "Queue Size",
    "avg_threads":    "Threads",
    "avg_e2e_latency": "E2E Latency",
    "violation_rate": "Violation Rate",
    "avg_cpu_threads": "CPU Threads",
    "cpu_usage_pct":  "CPU Usage %",
}

# Node-type palette (background fill, border)
_PALETTE = {
    "service":  ("#4C72B0", "#2c4f8c"),   # blue
    "workflow": ("#DD8452", "#b05e2a"),   # orange
    "node":     ("#55A868", "#2e7a45"),   # green
}


def friendly_label(col: str) -> str:
    """Convert a raw column name to a multi-line human-readable label."""
    if m := re.fullmatch(r"s(\d+)_(.+)", col):
        sid, metric = m.groups()
        return f"Service {sid}\n{_METRIC_NAMES.get(metric, metric.replace('_', ' ').title())}"
    if m := re.fullmatch(r"wf(\d+)_(.+)", col):
        wid, metric = m.groups()
        return f"Workflow {wid}\n{_METRIC_NAMES.get(metric, metric.replace('_', ' ').title())}"
    if m := re.fullmatch(r"node(\d+)_(.+)", col):
        nid, metric = m.groups()
        return f"Node {nid}\n{_METRIC_NAMES.get(metric, metric.replace('_', ' ').title())}"
    return col.replace("_", " ").title()


def _node_group(col: str) -> str:
    if col.startswith("s"):
        return "service"
    if col.startswith("wf"):
        return "workflow"
    return "node"


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _grouped_layout(G: nx.DiGraph, columns: list[str]) -> dict:
    """
    Place nodes in three vertical bands: services (left), workflows (centre),
    infrastructure nodes (right).  Within each band nodes are spread vertically.
    """
    groups = {"service": [], "workflow": [], "node": []}
    for col in columns:
        if col in G:
            groups[_node_group(col)].append(col)

    x_map = {"service": 0.0, "workflow": 0.5, "node": 1.0}
    pos = {}
    for grp, members in groups.items():
        x = x_map[grp]
        n = len(members)
        for i, node in enumerate(members):
            y = 1.0 - i / max(n - 1, 1)
            pos[node] = (x + np.random.uniform(-0.06, 0.06), y)
    return pos


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------

def plot_causal_graph(
    W: np.ndarray,
    columns: list[str],
    *,
    title: str = "Causal Graph",
    binary_threshold: float = 0.1,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Draw a beautified causal graph.

    Parameters
    ----------
    W               : weighted adjacency matrix (n × n)
    columns         : column names corresponding to rows/cols of W
    title           : figure title
    binary_threshold: edge weight magnitude below which edges are hidden
    save_path       : if given, save the figure to this path (PNG @ 180 dpi)
    show            : call plt.show() after drawing
    """
    B = get_binary_graph(W, binary_threshold)

    G = nx.DiGraph()
    labels = {col: friendly_label(col) for col in columns}
    G.add_nodes_from(columns)
    for i, j in zip(*np.nonzero(B)):
        G.add_edge(columns[i], columns[j], weight=float(W[i, j]))

    # Remove isolated nodes for a cleaner view
    isolated = [n for n in G.nodes if G.degree(n) == 0]
    G.remove_nodes_from(isolated)
    active_cols = [c for c in columns if c in G]

    pos = _grouped_layout(G, active_cols)

    # ---- Figure setup --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(18, 11))
    fig.patch.set_facecolor("#f7f7f7")
    ax.set_facecolor("#f7f7f7")

    # ---- Nodes ---------------------------------------------------------------
    for grp, (fill, border) in _PALETTE.items():
        members = [n for n in active_cols if _node_group(n) == grp and n in G]
        if not members:
            continue
        nx.draw_networkx_nodes(
            G, pos, nodelist=members, ax=ax,
            node_size=2200, node_color=fill, edgecolors=border, linewidths=2.0,
            alpha=0.95,
        )

    # ---- Labels (friendly, white) --------------------------------------------
    nx.draw_networkx_labels(
        G, pos, labels={n: labels[n] for n in G.nodes()}, ax=ax,
        font_size=7, font_color="white", font_weight="bold",
    )

    # ---- Edges ---------------------------------------------------------------
    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] >= 0]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < 0]

    shared_edge_kw = dict(ax=ax, arrows=True, arrowsize=18,
                          connectionstyle="arc3,rad=0.12", min_source_margin=28,
                          min_target_margin=28)
    if pos_edges:
        nx.draw_networkx_edges(G, pos, edgelist=pos_edges, edge_color="#2ca02c",
                               width=2.0, **shared_edge_kw)
    if neg_edges:
        nx.draw_networkx_edges(G, pos, edgelist=neg_edges, edge_color="#d62728",
                               width=2.0, style="dashed", **shared_edge_kw)

    # ---- Edge weight labels --------------------------------------------------
    edge_labels = {(u, v): f"{d['weight']:+.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, ax=ax,
        font_size=6.5, font_color="#333333",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.65, ec="none"),
    )

    # ---- Legend --------------------------------------------------------------
    legend_handles = [
        mpatches.Patch(color=fill, label=grp.title())
        for grp, (fill, _) in _PALETTE.items()
    ] + [
        mpatches.Patch(color="#2ca02c", label="Positive causal effect"),
        mpatches.Patch(color="#d62728", label="Negative causal effect",
                       linestyle="dashed"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.85,
              fontsize=9, title="Legend", title_fontsize=9)

    # ---- Band labels ---------------------------------------------------------
    for label, x in [("Services", 0.0), ("Workflows", 0.5), ("Infra Nodes", 1.0)]:
        ax.text(x, 1.06, label, transform=ax.transData, ha="center", va="bottom",
                fontsize=10, color="#555555", fontstyle="italic")

    n_edges = len(G.edges())
    n_nodes = len(G.nodes())
    ax.set_title(f"{title}  —  {n_nodes} variables, {n_edges} edges",
                 fontsize=14, fontweight="bold", pad=18)
    ax.axis("off")
    fig.tight_layout(pad=1.5)

    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Graph plot saved: {save_path}")

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import pandas as pd
    from causal_discovery import causal_discovery

    parser = argparse.ArgumentParser(description="Plot causal graph from CSV")
    parser.add_argument("csv", help="Path to CSV dataset")
    parser.add_argument("--lambda1", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--save", default=None, help="Output PNG path")
    args = parser.parse_args()

    W, columns = causal_discovery(args.csv, lambda1=args.lambda1,
                                  w_threshold=args.threshold)
    plot_causal_graph(W, columns, binary_threshold=args.threshold,
                      save_path=args.save or Path(args.csv).with_suffix(".png"))
