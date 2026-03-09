import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from mess3_common import (
    Mess3,
    Mess3Mixture,
    GPT,
    get_residuals_and_beliefs,
    SEQ_LEN,
    VOCAB_SIZE,
    D_MODEL,
    N_HEADS,
    N_LAYERS,
)


# Core CEV helpers
def cev(activations):
    """
    Compute cumulative explained variance for a set of activation vectors.

    parameters
        activations : (N, D) array

    returns
        cev_curve : (D,) array, CEV(k) for k = 1..D
        eigenvalues : (D,) array, individual explained variance fractions
    """
    acts_c = activations - activations.mean(axis = 0)
    cov = acts_c.T @ acts_c / len(acts_c)
    eigvals = np.linalg.eigvalsh(cov)[::-1]  # descending
    eigvals = np.maximum(eigvals, 0.0)  # numerical safety
    total = eigvals.sum()
    if total < 1e-12:
        return np.zeros(len(eigvals)), np.zeros(len(eigvals))
    fracs = eigvals / total
    return np.cumsum(fracs), fracs


def effective_dim(activations, threshold = 0.95):
    """
    k* = smallest k such that CEV(k) >= threshold.
    returns int.
    """
    cev_curve, _ = cev(activations)
    hits = np.where(cev_curve >= threshold)[0]
    return int(hits[0] + 1) if len(hits) else len(cev_curve)


# full CEV curves at selected positions
def plot_cev_curves_by_position(acts, positions, tag = ""):
    """
    Plot full CEV(k) curves for all context positions on one axes.
    """
    T_vals = np.unique(positions)
    cmap = plt.cm.viridis
    colors = [cmap(v) for v in np.linspace(0.0, 1.0, len(T_vals))]

    fig, ax = plt.subplots(figsize = (7, 4))

    max_k = 0
    for t, col in zip(T_vals, colors):
        mask_t = positions == t
        cev_t, _ = cev(acts[mask_t])
        k_range = np.arange(1, len(cev_t) + 1)
        max_k = max(max_k, len(cev_t))
        ax.plot(k_range, cev_t, color = col, lw = 1.5, alpha = 0.85)

    # colorbar to show t mapping
    sm = plt.cm.ScalarMappable(
        cmap = cmap,
        norm = plt.Normalize(vmin = T_vals.min(), vmax = T_vals.max()),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax = ax, pad = 0.02)
    cbar.set_label("Context position t", fontsize = 9)

    ax.axhline(0.95, ls = "--", color = "gray", lw = 1, label = "95% threshold")
    ax.axvline(5, ls = ":", color = "#2dc653", lw = 1.5, label = "d = 5 (predicted factored)")

    ax.set(
        xlabel = "Number of principal components k",
        ylabel = "Cumulative explained variance",
        title = "CEV curves at every context position",
        xlim = (1, min(max_k, 40)),
        ylim = (0, 1.05),
    )
    ax.legend(fontsize = 8, loc = "lower right")
    ax.grid(alpha = 0.2)
    plt.tight_layout()

    fname = f"cev_curves_by_position{tag}.png"
    plt.savefig(fname, dpi = 150, bbox_inches = "tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type = str, default = "mess3_ckpt.pt")
    parser.add_argument("--n_seqs", type = int, default = 3000)
    parser.add_argument("--threshold", type = float, default = 0.95)
    parser.add_argument("--seed", type = int, default = 0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location = device)

    comp_A = Mess3(alpha = ckpt["comp_A_alpha"], x = ckpt["comp_A_x"])
    comp_B = Mess3(alpha = ckpt["comp_B_alpha"], x = ckpt["comp_B_x"])
    mixture = Mess3Mixture(comp_A, comp_B, mixing_prob = 0.5)
    seq_len = ckpt.get("seq_len", SEQ_LEN)

    model = GPT(
        vocab_size = ckpt.get("vocab_size", VOCAB_SIZE),
        d_model = ckpt.get("d_model", D_MODEL),
        n_heads = ckpt.get("n_heads", N_HEADS),
        n_layers = ckpt.get("n_layers", N_LAYERS),
        max_seq_len = seq_len,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    acts, bA, bB, labels, positions, posteriors = get_residuals_and_beliefs(
        model,
        mixture,
        n_seqs = args.n_seqs,
        seq_len = seq_len,
        layer_idx = -1,
        device = device,
    )

    tag = "_lastblock"

    plot_cev_curves_by_position(acts, positions, tag = tag)

    return {
        "beliefs_A": bA,
        "beliefs_B": bB,
        "posteriors": posteriors,
    }


if __name__ == "__main__":
    results = main()