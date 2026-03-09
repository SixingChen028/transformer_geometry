import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

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


# Probe helpers
def fit_probe(activations, targets):
    A = np.hstack([activations, np.ones((len(activations), 1))])
    W, *_ = np.linalg.lstsq(A, targets, rcond = None)
    return W


def apply_probe(activations, W):
    A = np.hstack([activations, np.ones((len(activations), 1))])
    return A @ W


def r2_score(true, pred):
    r2s = []
    for s in range(true.shape[1]):
        ss_res = ((true[:, s] - pred[:, s]) ** 2).sum()
        ss_tot = ((true[:, s] - true[:, s].mean()) ** 2).sum()
        r2s.append(float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0)
    return float(np.mean(r2s))


def get_subspace_basis(W, d_model, n_components):
    """
    top-n left singular vectors of probe weight matrix (bias dropped).
    """
    U, _, _ = np.linalg.svd(W[:d_model], full_matrices = False)
    return U[:, :n_components]


def projection_fraction(v, Q):
    """
    Fraction of the norm of vector v explained by projection onto the column space of Q (orthonormal basis).
    = ||Q Q^T v||^2 / ||v||^2
    """
    v = v / (np.linalg.norm(v) + 1e-12)
    proj = Q @ (Q.T @ v)
    return float(np.dot(proj, proj))


def orthogonal_complement_projection(activations, Q):
    """
    Project activations onto the orthogonal complement of span(Q).
    returns residual activations of same shape.
    """
    # proj onto Q: A Q Q^T, residual: A - A Q Q^T
    coeff = activations @ Q  # (N, k)
    return activations - coeff @ Q.T  # (N, D)


# projection fraction of post_A onto span(Q_A \cup Q_B)
def diagnostic_projection_fraction(W_A, W_B, W_post, d_model, n_components = 2, n_shuffles = 500, seed = 0, tag = ""):
    """
    Project the post_A probe direction onto span(Q_A \cup Q_B) and measure the fraction of its norm recovered.

    Null: same computation with permuted probe directions (random W_post).
    """
    Q_A = get_subspace_basis(W_A, d_model, n_components)
    Q_B = get_subspace_basis(W_B, d_model, n_components)
    Q_post = get_subspace_basis(W_post, d_model, 1)[:, 0]  # 1D direction

    # orthonormal basis for span(Q_A ∪ Q_B) via QR
    Q_union, _ = np.linalg.qr(np.hstack([Q_A, Q_B]))
    Q_union = Q_union[:, :2 * n_components]  # at most 4D

    actual_frac = projection_fraction(Q_post, Q_union)

    # null: random unit vectors in R^d_model
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n_shuffles):
        v_rand = rng.standard_normal(d_model)
        v_rand /= np.linalg.norm(v_rand)
        nulls.append(projection_fraction(v_rand, Q_union))
    nulls = np.array(nulls)
    null_mean = nulls.mean()
    null_std = nulls.std()
    z = (actual_frac - null_mean) / (null_std + 1e-12)

    geometric_baseline = (2 * n_components) / d_model

    fig, ax = plt.subplots(figsize = (5, 4))
    ax.boxplot(
        nulls,
        positions = [0],
        widths = 0.4,
        patch_artist = True,
        boxprops = dict(facecolor = "#adb5bd", alpha = 0.6),
        medianprops = dict(color = "black", lw = 2),
        flierprops = dict(marker = "o", markersize = 3, alpha = 0.3),
    )
    ax.scatter([0], [actual_frac], color = "#e63946", s = 120, zorder = 5, marker = "D", label = f"post_A  (z = {z:.1f})")
    ax.axhline(
        geometric_baseline,
        ls = ":",
        color = "gray",
        lw = 1.2,
        label = f"Geometric baseline",
    )
    ax.set_xticks([0])
    ax.set_xticklabels([""])
    ax.set_ylabel("Fraction of norm in span(Q_A U Q_B)")
    ax.set_title(
        "Is post_A in the span of the belief subspaces?\n"
        "High fraction = post_A is derived from belief states",
    )
    ax.legend(fontsize = 8)
    ax.grid(axis = "y", alpha = 0.3)
    plt.tight_layout()

    fname = f"diag1_projection_fraction{tag}.png"
    plt.savefig(fname, dpi = 150, bbox_inches = "tight")
    plt.close()

    return actual_frac, null_mean, z


# residual decoding R^2
def diagnostic_residual_decoding(activations, beliefs_A, beliefs_B, posteriors, W_A, W_B, d_model, n_components = 2, tag = ""):
    """
    Remove span(Q_A \cup Q_B) from activations, then fit a fresh post_A probe in the residual space.
    """
    Q_A = get_subspace_basis(W_A, d_model, n_components)
    Q_B = get_subspace_basis(W_B, d_model, n_components)
    Q_union, _ = np.linalg.qr(np.hstack([Q_A, Q_B]))
    Q_union = Q_union[:, :2 * n_components]

    post_targets = posteriors.reshape(-1, 1)

    # full-space R^2
    W_post_full = fit_probe(activations, post_targets)
    r2_full = r2_score(post_targets, apply_probe(activations, W_post_full))

    # residual-space R^2
    acts_resid = orthogonal_complement_projection(activations, Q_union)
    W_post_resid = fit_probe(acts_resid, post_targets)
    r2_resid = r2_score(post_targets, apply_probe(acts_resid, W_post_resid))

    # also: R^2 from span(Q_A ∪ Q_B) alone
    acts_proj = activations - acts_resid  # projection onto span
    W_post_proj = fit_probe(acts_proj, post_targets)
    r2_proj = r2_score(post_targets, apply_probe(acts_proj, W_post_proj))

    labels = ["Full space", "span(Q_A,Q_B) only", "Orthogonal\ncomplement"]
    values = [r2_full, r2_proj, r2_resid]
    colors = ["#333333", "#4361ee", "#e63946"]

    fig, ax = plt.subplots(figsize = (5, 4))
    bars = ax.bar(labels, values, color = colors, alpha = 0.8, width = 0.5)
    ax.set_ylabel("post_A decoding R^2")
    ax.set_title(
        "Can post_A be decoded from the\n"
        "orthogonal complement of span(eta_A, eta_B)?",
    )
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0, ls = "--", color = "gray", lw = 1)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha = "center", va = "bottom", fontsize = 10)
    ax.grid(axis = "y", alpha = 0.3)
    plt.tight_layout()

    fname = f"diag2_residual_decoding{tag}.png"
    plt.savefig(fname, dpi = 150, bbox_inches = "tight")
    plt.close()

    return r2_full, r2_proj, r2_resid



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
    seq_len = ckpt.get("seq_len", SEQ_LEN)
    d_model = ckpt.get("d_model", D_MODEL)

    comp_A = Mess3(alpha = ckpt["comp_A_alpha"], x = ckpt["comp_A_x"])
    comp_B = Mess3(alpha = ckpt["comp_B_alpha"], x = ckpt["comp_B_x"])
    mixture = Mess3Mixture(comp_A, comp_B, mixing_prob = 0.5)

    model = GPT(
        vocab_size = ckpt.get("vocab_size", VOCAB_SIZE),
        d_model = d_model,
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

    # fit probes on all sequences (consistent with probe analysis)
    W_A = fit_probe(acts, bA)
    W_B = fit_probe(acts, bB)
    W_post = fit_probe(acts, posteriors.reshape(-1, 1))

    diagnostic_projection_fraction(
        W_A,
        W_B,
        W_post,
        d_model,
        n_components = 2,
        n_shuffles = 500,
        seed = args.seed,
        tag = tag,
    )

    diagnostic_residual_decoding(
        acts,
        bA,
        bB,
        posteriors,
        W_A,
        W_B,
        d_model,
        n_components = 2,
        tag = tag,
    )


if __name__ == "__main__":
    main()