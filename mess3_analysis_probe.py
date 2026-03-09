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

STATE_COLORS = ["#e41a1c", "#377eb8", "#4daf4a"]  # hidden states 0, 1, 2


# Probe helpers
def fit_probe(activations, targets):
    """
    OLS: [acts | 1] @ W = targets.
    returns W of shape (D + 1, n_targets).
    """
    A = np.hstack([activations, np.ones((len(activations), 1))])
    W, *_ = np.linalg.lstsq(A, targets, rcond = None)
    return W


def apply_probe(activations, W):
    A = np.hstack([activations, np.ones((len(activations), 1))])
    return A @ W


def r2_score(true, pred):
    """
    mean R^2 across all target dimensions.
    """
    r2s = []
    for s in range(true.shape[1]):
        ss_res = ((true[:, s] - pred[:, s]) ** 2).sum()
        ss_tot = ((true[:, s] - true[:, s].mean()) ** 2).sum()
        r2s.append(float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0)
    return float(np.mean(r2s))


# Belief decoding
def plot_belief_decoding(activations, beliefs_A, beliefs_B, labels, positions, posteriors, tag = ""):
    """
    Fit W_A and W_B on all sequences (both components), then evaluate per-position R^2 by applying these fixed probes at each t.
    returns dict with probe weight matrices W_A, W_B, W_post.
    """
    T_vals = np.unique(positions)

    # fit probes on all sequences
    W_A = fit_probe(activations, beliefs_A)
    W_B = fit_probe(activations, beliefs_B)
    W_post = fit_probe(activations, posteriors.reshape(-1, 1))

    dec_A = apply_probe(activations, W_A)
    dec_B = apply_probe(activations, W_B)
    dec_post = apply_probe(activations, W_post)

    # per-component figures
    for ci, label_str in [(0, "A"), (1, "B")]:
        beliefs_ci = beliefs_A if ci == 0 else beliefs_B
        W = W_A if ci == 0 else W_B
        mask_ci = labels == ci

        # PCA on ground-truth beliefs (all positions) for consistent 2D axes
        pca_bel = PCA(n_components = 2).fit(beliefs_ci)
        gt_all = pca_bel.transform(beliefs_ci[mask_ci])
        pad = 0.08 * max(
            gt_all[:, 0].max() - gt_all[:, 0].min(),
            gt_all[:, 1].max() - gt_all[:, 1].min(),
        )
        xlim = (gt_all[:, 0].min() - pad, gt_all[:, 0].max() + pad)
        ylim = (gt_all[:, 1].min() - pad, gt_all[:, 1].max() + pad)

        n_pos = len(T_vals)
        r2_per_pos = []

        fig, axes = plt.subplots(2, n_pos, figsize = (3.0 * n_pos, 5.5))
        fig.suptitle(
            f"Component {label_str} — probe fit on all sequences\n"
            f"Row 0: ground truth   Row 1: decoded from residual stream",
            fontsize = 10,
            y = 1.02,
        )

        if n_pos == 1:
            axes = np.array(axes).reshape(2, 1)

        for col, t in enumerate(T_vals):
            # evaluate only on true-component sequences at this position
            pos_mask = mask_ci & (positions == t)
            if pos_mask.sum() < 10:
                for row in range(2):
                    axes[row, col].set_visible(False)
                r2_per_pos.append(np.nan)
                continue

            bels_t = beliefs_ci[pos_mask]
            acts_t = activations[pos_mask]
            dom_t = bels_t.argmax(axis = 1)

            dec_t = apply_probe(acts_t, W)
            r2_t = r2_score(bels_t, dec_t)
            r2_per_pos.append(r2_t)

            gt_2d = pca_bel.transform(bels_t)
            dec_2d = pca_bel.transform(np.clip(dec_t, 0, 1))

            # row 0: ground-truth
            ax = axes[0, col]
            for s in range(3):
                m = dom_t == s
                if m.sum():
                    ax.scatter(
                        gt_2d[m, 0],
                        gt_2d[m, 1],
                        c = STATE_COLORS[s],
                        s = 8,
                        alpha = 0.55,
                        label = f"s{s}",
                    )
            ax.set(xlim = xlim, ylim = ylim, title = f"t = {t}")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel("Ground truth", fontsize = 10)
                ax.legend(fontsize = 6, markerscale = 2, loc = "upper left")

            # row 1: decoded
            ax2 = axes[1, col]
            ax2.scatter(gt_2d[:, 0], gt_2d[:, 1], c = "lightgray", s = 3, alpha = 0.25, zorder = 0)
            for s in range(3):
                m = dom_t == s
                if m.sum():
                    ax2.scatter(
                        dec_2d[m, 0],
                        dec_2d[m, 1],
                        c = STATE_COLORS[s],
                        s = 8,
                        alpha = 0.55,
                    )
            ax2.set(xlim = xlim, ylim = ylim, title = f"R^2 = {r2_t:.3f}")
            ax2.set_xticks([])
            ax2.set_yticks([])
            if col == 0:
                ax2.set_ylabel("Decoded", fontsize = 10)

        plt.tight_layout()
        fname = f"belief_decoding_comp{label_str}{tag}.png"
        plt.savefig(fname, dpi = 150, bbox_inches = "tight")
        plt.close()

        # R^2 over position
        fig2, ax3 = plt.subplots(figsize = (6, 3.2))
        ax3.plot(T_vals, r2_per_pos, "o-", color = "#6a0572", lw = 2, ms = 5)
        ax3.axhline(1.0, ls = "--", color = "gray", lw = 1, label = "Perfect")
        ax3.set(
            xlabel = "Context position t",
            ylabel = "Mean R^2",
            title = f"Component {label_str}: R^2 over context (probe fit on all sequences)",
            ylim = (0, 1.05),
            xticks = T_vals,
        )
        ax3.grid(axis = "y", alpha = 0.3)
        ax3.legend(fontsize = 9)
        plt.tight_layout()
        fname2 = f"r2_over_position_comp{label_str}{tag}.png"
        plt.savefig(fname2, dpi = 150, bbox_inches = "tight")
        plt.close()

    # post_A R^2 over position
    r2_post_per_pos = []
    for t in T_vals:
        pos_mask = positions == t
        if pos_mask.sum() < 10:
            r2_post_per_pos.append(np.nan)
            continue
        post_t = posteriors[pos_mask].reshape(-1, 1)
        acts_t = activations[pos_mask]
        dec_t = apply_probe(acts_t, W_post)
        r2_post_per_pos.append(r2_score(post_t, dec_t))

    fig3, ax4 = plt.subplots(figsize = (6, 3.2))
    ax4.plot(T_vals, r2_post_per_pos, "o-", color = "#2dc653", lw = 2, ms = 5)
    ax4.axhline(1.0, ls = "--", color = "gray", lw = 1, label = "Perfect")
    ax4.set(
        xlabel = "Context position t",
        ylabel = "Mean R^2",
        title = "post_A: R^2 over context position\n(probe fit on all sequences)",
        ylim = (0, 1.05),
        xticks = T_vals,
    )
    ax4.grid(axis = "y", alpha = 0.3)
    ax4.legend(fontsize = 9)
    plt.tight_layout()
    fname3 = f"r2_over_position_post{tag}.png"
    plt.savefig(fname3, dpi = 150, bbox_inches = "tight")
    plt.close()

    return {"A": W_A, "B": W_B, "post": W_post}


# Subspace orthogonality
def get_subspace_basis(W, d_model, n_components):
    """
    Extract orthonormal basis for the subspace spanned by the probe directions.
    W has shape (D + 1, n_targets); drop the bias row, then take top n_components left singular vectors.
    returns Q of shape (D, n_components).
    """
    U, _, _ = np.linalg.svd(W[:d_model], full_matrices = False)
    return U[:, :n_components]


def subspace_overlap(Q1, Q2):
    """
    Overlap between subspaces with orthonormal bases Q1, Q2 = (1 / d_min) * ||Q1.T @ Q2||_F^2
    0 = orthogonal, 1 = identical.
    """
    M = Q1.T @ Q2
    d_min = min(Q1.shape[1], Q2.shape[1])
    return float(np.sum(M ** 2) / d_min)


def compute_null_distribution(activations, beliefs_A, beliefs_B, posteriors, d_model, n_components = 2, n_shuffles = 500, seed = 0):
    """
    Null distribution by randomly rotating probe directions.
    At each iteration we generate random weight matrices of the same shape as the real probes (by fitting probes to permuted targets), extract
    subspace bases, and compute pairwise overlaps.
    returns array (n_shuffles, 3): columns are overlap(A,B), overlap(A,post), overlap(B,post)
    """
    rng = np.random.default_rng(seed)
    nulls = np.zeros((n_shuffles, 3))

    for i in range(n_shuffles):
        perm = rng.permutation(len(beliefs_A))
        W_A_r = fit_probe(activations, beliefs_A[perm])
        W_B_r = fit_probe(activations, beliefs_B[perm])
        W_p_r = fit_probe(activations, posteriors[perm].reshape(-1, 1))

        Q_A = get_subspace_basis(W_A_r, d_model, n_components)
        Q_B = get_subspace_basis(W_B_r, d_model, n_components)
        Q_p = get_subspace_basis(W_p_r, d_model, 1)

        nulls[i, 0] = subspace_overlap(Q_A, Q_B)
        nulls[i, 1] = subspace_overlap(Q_A, Q_p)
        nulls[i, 2] = subspace_overlap(Q_B, Q_p)

    return nulls


def plot_orthogonality(probe_weights, activations, beliefs_A, beliefs_B, posteriors, d_model, n_components = 2, n_shuffles = 500, tag = ""):
    """
    Extract orthonormal bases Q_A (2D), Q_B (2D), Q_post (1D) from probes.
    Compute pairwise subspace overlaps.
    Compare against null distribution from permuted probes.
    Plot bar chart with null boxplots and actual overlaps overlaid.
    """
    Q_A = get_subspace_basis(probe_weights["A"], d_model, n_components)
    Q_B = get_subspace_basis(probe_weights["B"], d_model, n_components)
    Q_post = get_subspace_basis(probe_weights["post"], d_model, 1)

    actual = np.array([
        subspace_overlap(Q_A, Q_B),
        subspace_overlap(Q_A, Q_post),
        subspace_overlap(Q_B, Q_post),
    ])
    pair_labels = ["eta_A vs eta_B", "eta_A vs post", "eta_B vs post"]

    nulls = compute_null_distribution(
        activations,
        beliefs_A,
        beliefs_B,
        posteriors,
        d_model,
        n_components = n_components,
        n_shuffles = n_shuffles,
    )

    null_mean = nulls.mean(axis = 0)
    null_std = nulls.std(axis = 0)
    z_scores = (actual - null_mean) / (null_std + 1e-12)

    x = np.arange(len(pair_labels))
    fig, ax = plt.subplots(figsize = (7, 4.5))

    ax.boxplot(
        [nulls[:, i] for i in range(3)],
        positions = x,
        widths = 0.4,
        patch_artist = True,
        boxprops = dict(facecolor = "#adb5bd", alpha = 0.6),
        medianprops = dict(color = "black", lw = 2),
        whiskerprops = dict(lw = 1.2),
        capprops = dict(lw = 1.2),
        flierprops = dict(marker = "o", markersize = 3, alpha = 0.3),
    )

    ax.scatter(x, actual, color = "#e63946", s = 100, zorder = 5, label = "Actual", marker = "D")

    for xi, (v, z) in enumerate(zip(actual, z_scores)):
        ax.text(xi, v + 0.003, f"z = {z:.1f}", ha = "center", va = "bottom", fontsize = 9, color = "#e63946")

    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize = 10)
    ax.set_ylabel("Subspace overlap (0 = orthogonal, 1 = identical)", fontsize = 10)
    ax.set_title(
        "Subspace orthogonality: eta_A, eta_B, posterior\n"
        "Gray boxes = null distribution from permuted probes",
        fontsize = 10,
    )
    ax.set_ylim(0, max(nulls.max(), actual.max()) * 1.2)
    ax.legend(fontsize = 9)
    ax.grid(axis = "y", alpha = 0.3)
    plt.tight_layout()

    fname = f"subspace_orthogonality{tag}.png"
    plt.savefig(fname, dpi = 150, bbox_inches = "tight")
    plt.close()

    return {
        "actual": actual,
        "null_mean": null_mean,
        "null_std": null_std,
        "z_scores": z_scores,
    }



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

    probe_weights = plot_belief_decoding(
        acts,
        bA,
        bB,
        labels,
        positions,
        posteriors,
        tag = "_lastblock",
    )

    orth_stats = plot_orthogonality(
        probe_weights,
        acts,
        bA,
        bB,
        posteriors,
        d_model = ckpt.get("d_model", D_MODEL),
        n_components = 2,
        n_shuffles = 500,
        tag = "_lastblock",
    )

    return {
        "probe_weights": probe_weights,
        "orth_stats": orth_stats,
    }


if __name__ == "__main__":
    results = main()