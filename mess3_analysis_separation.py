import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

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

COMP_COLORS = ["#f77f00", "#4361ee"]  # A = orange, B = blue
COMP_LABELS = ["Comp A (alpha = 0.95)", "Comp B (alpha = 0.05)"]


# PCA scatter by component, one panel per position

def plot_pca_by_position(acts, labels, positions, tag = ""):
    """
    each panel: PC1-PC2 scatter of residual stream activations at that position, colored by true component label.
    PCA is fit once on all activations so axes are comparable across panels.
    """
    T_vals = np.unique(positions)
    n_pos = len(T_vals)

    # fit PCA on all activations once
    pca = PCA(n_components = 2)
    proj = pca.fit_transform(acts)  # (N*T, 2)

    # shared axis limits
    pad = 0.08 * max(
        proj[:, 0].max() - proj[:, 0].min(),
        proj[:, 1].max() - proj[:, 1].min(),
    )
    xlim = (proj[:, 0].min() - pad, proj[:, 0].max() + pad)
    ylim = (proj[:, 1].min() - pad, proj[:, 1].max() + pad)

    # subsample for plotting speed (max 400 points per position)
    rng = np.random.default_rng(0)
    n_plot = 400

    fig, axes = plt.subplots(1, n_pos, figsize = (2.8 * n_pos, 3.2), sharey = True)
    fig.suptitle(
        "Residual stream PCA colored by component label",
        fontsize = 11,
    )

    if n_pos == 1:
        axes = [axes]

    for col, t in enumerate(T_vals):
        ax = axes[col]
        mask = positions == t
        p_t = proj[mask]
        l_t = labels[mask]

        for ci in [0, 1]:
            idx = np.where(l_t == ci)[0]
            if len(idx) > n_plot:
                idx = rng.choice(idx, n_plot, replace = False)
            ax.scatter(
                p_t[idx, 0],
                p_t[idx, 1],
                c = COMP_COLORS[ci],
                s = 5,
                alpha = 0.45,
                label = COMP_LABELS[ci] if col == 0 else None,
            )

        ax.set(xlim = xlim, ylim = ylim, title = f"t = {t}")
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("PC 2", fontsize = 9)
            ax.legend(fontsize = 7, markerscale = 2, loc = "upper left")
        ax.set_xlabel("PC 1", fontsize = 9)

    plt.tight_layout()
    fname = f"pca_by_position{tag}.png"
    plt.savefig(fname, dpi = 150, bbox_inches = "tight")
    plt.close()


# linear separability (logistic regression CV) over position
def plot_separability_over_position(acts, labels, positions, tag = ""):
    """
    At each context position t, fit a logistic regression from residual stream activations to component label using 5-fold cross-validation.
    Plot the mean CV accuracy +- 1 std as a function of t.
    Chance = 0.5. A well-trained model should approach 1.0 at late t.
    """
    T_vals = np.unique(positions)
    accs = []
    stds = []

    for t in T_vals:
        mask = positions == t
        X_t = acts[mask]
        y_t = labels[mask]
        clf = LogisticRegression(
            max_iter = 500,
            C = 1.0,
            solver = "lbfgs",
            multi_class = "auto",
        )
        cv = cross_val_score(clf, X_t, y_t, cv = 5, scoring = "accuracy")
        accs.append(cv.mean())
        stds.append(cv.std())

    accs = np.array(accs)
    stds = np.array(stds)

    fig, ax = plt.subplots(figsize = (6, 3.5))
    ax.plot(T_vals, accs, "o-", color = "#2d6a4f", lw = 2, ms = 5, label = "CV accuracy")
    ax.fill_between(T_vals, accs - stds, accs + stds, alpha = 0.2, color = "#2d6a4f")
    ax.axhline(0.5, ls = "--", color = "gray", lw = 1, label = "Chance")
    ax.set(
        xlabel = "Context position t",
        ylabel = "Component classification accuracy",
        title = "Linear separability of components grows with context",
        ylim = (0.4, 1.05),
        xticks = T_vals,
    )
    ax.legend(fontsize = 9)
    ax.grid(axis = "y", alpha = 0.3)
    plt.tight_layout()

    fname = f"separability_over_position{tag}.png"
    plt.savefig(fname, dpi = 150, bbox_inches = "tight")
    plt.close()

    return accs, stds



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type = str, default = "mess3_ckpt.pt")
    parser.add_argument("--n_seqs", type = int, default = 3000)
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
    plot_pca_by_position(acts, labels, positions, tag = tag)
    accs, stds = plot_separability_over_position(acts, labels, positions, tag = tag)

    return {
        "accs": accs,
        "stds": stds,
        "beliefs_A": bA,
        "beliefs_B": bB,
    }


if __name__ == "__main__":
    results = main()