"""Causal discovery using NOTEARS (Zheng et al. 2018, NeurIPS).

Self-contained implementation — no causalnex/causal-learn dependency required.
Only needs numpy and scipy (already in requirements.txt).
"""

import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# NOTEARS core (linear, continuous acyclicity constraint)
# ---------------------------------------------------------------------------

def _notears_linear(X: np.ndarray, lambda1: float, max_iter: int,
                    h_tol: float, rho_max: float) -> np.ndarray:
    """
    Solve the NOTEARS optimisation problem (Zheng et al. 2018).

    Minimises  0.5/n ||X - X W||²_F  +  lambda1 * ||W||_1
    subject to  h(W) = tr(e^(W◦W)) - d = 0  (acyclicity).

    Returns the (d, d) weighted adjacency matrix W.
    """
    n, d = X.shape
    rho, alpha, h_prev = 1.0, 0.0, np.inf

    def _h(W: np.ndarray) -> float:
        """DAG acyclicity measure (equals 0 iff W is a DAG)."""
        return float(np.trace(np.linalg.matrix_power(
            np.eye(d) + W * W / d, d)) - d)

    def _obj(w_vec: np.ndarray):
        W = w_vec.reshape(d, d)
        # Least-squares loss
        R = X - X @ W
        loss = 0.5 / n * (R * R).sum()
        G_loss = -1.0 / n * X.T @ R
        # Acyclicity penalty
        h_val = _h(W)
        penalty = 0.5 * rho * h_val * h_val + alpha * h_val
        # Gradient of h w.r.t. W:  2 * (I + W◦W/d)^(d-1) ◦ W  *  (d/d) ... simplified
        E = np.linalg.matrix_power(np.eye(d) + W * W / d, d - 1)
        G_h = (2.0 / d) * E.T * W
        G_penalty = (rho * h_val + alpha) * G_h
        # L1 proximal is handled via w+ / w- splitting
        obj = loss + penalty
        grad = (G_loss + G_penalty).flatten()
        return obj, grad

    # Split W = W+ - W- (both >= 0) for L1 via bound constraints
    w_est = np.zeros(2 * d * d)
    bounds = [(0, None)] * (2 * d * d)

    for _ in range(max_iter):
        def _aug(w_vec):
            # W = W+ - W-
            Wp = w_vec[:d*d].reshape(d, d)
            Wm = w_vec[d*d:].reshape(d, d)
            W = Wp - Wm
            obj, grad = _obj(W.flatten())
            obj += lambda1 * w_vec.sum()
            grad_full = np.concatenate([grad + lambda1, -grad + lambda1])
            return obj, grad_full

        res = minimize(_aug, w_est, method="L-BFGS-B", jac=True, bounds=bounds,
                       options={"maxiter": 1000, "ftol": 1e-8, "gtol": 1e-8})
        w_est = res.x
        Wp = w_est[:d*d].reshape(d, d)
        Wm = w_est[d*d:].reshape(d, d)
        W = Wp - Wm
        h_val = _h(W)
        if h_val > 0.25 * h_prev:
            rho = min(rho * 10, rho_max)
        alpha += rho * h_val
        h_prev = h_val
        if h_val <= h_tol or rho >= rho_max:
            break

    return W


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def causal_discovery(
    csv_path: str,
    lambda1: float = 0.1,
    w_threshold: float = 0.1,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    rho_max: float = 1e16,
) -> tuple:
    """
    Compute a causal graph from a CSV file using NOTEARS.

    Args:
        csv_path:    Path to the CSV dataset.
        lambda1:     L1 regularisation strength (higher = sparser graph).
        w_threshold: Edges with |weight| <= this are zeroed out post-solve.
        max_iter:    Maximum augmented-Lagrangian outer iterations.
        h_tol:       Acyclicity tolerance for early stopping.
        rho_max:     Maximum penalty coefficient.

    Returns:
        W:       Weighted adjacency matrix, shape (n_cols, n_cols).
                 W[i, j] != 0 means column i is a direct cause of column j.
        columns: List of column names (row/column index for W).
    """
    df = pd.read_csv(csv_path)

    # Remove constant columns — zero variance breaks standardisation and NOTEARS
    df = df.loc[:, df.std() > 1e-8]
    df = df.dropna()

    if df.empty or df.shape[1] < 2:
        raise ValueError(
            f"Dataset at '{csv_path}' has fewer than 2 non-constant "
            "columns after cleaning."
        )

    # Standardise so NOTEARS operates on comparable scales
    df = (df - df.mean()) / df.std()
    columns = list(df.columns)
    X = df.values.astype(float)

    W = _notears_linear(X, lambda1=lambda1, max_iter=max_iter,
                        h_tol=h_tol, rho_max=rho_max)
    W[np.abs(W) <= w_threshold] = 0.0

    return W, columns


def get_binary_graph(W: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Return binary adjacency matrix by thresholding absolute edge weights."""
    return (np.abs(W) > threshold).astype(int)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NOTEARS causal discovery from CSV")
    parser.add_argument("csv", help="Path to CSV dataset")
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--w-threshold", type=float, default=0.0)
    parser.add_argument("--binary-threshold", type=float, default=0.1)
    args = parser.parse_args()

    W, columns = causal_discovery(
        args.csv, beta=args.beta, w_threshold=args.w_threshold
    )

    print(f"Columns ({len(columns)}): {columns}")
    print("\nWeighted adjacency matrix W  (W[i,j] = i → j):")
    print(np.round(W, 3))

    B = get_binary_graph(W, args.binary_threshold)
    edges = [(columns[i], columns[j]) for i, j in zip(*np.nonzero(B))]
    print(f"\nBinary graph (threshold={args.binary_threshold}), {len(edges)} edges:")
    for src, dst in edges:
        print(f"  {src} → {dst}")
