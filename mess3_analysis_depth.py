import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from mess3_common import (
    Mess3,
    Mess3Mixture,
    GPT,
    SEQ_LEN,
    VOCAB_SIZE,
    D_MODEL,
    N_HEADS,
    N_LAYERS,
)


# Data collection: all layers in one forward pass
def collect_all_layers(model, mixture, n_seqs = 10000, seq_len = SEQ_LEN, device = "cpu"):
    """
    Run one forward pass and return residual stream activations at every layer alongside Bayesian quantities at every (sequence, position).

    returns
        all_acts: list of (N*T, D) arrays, length = n_layers + 1
            index 0 = embedding, index k = after block k
        beliefs_A: (N*T, 3)
        beliefs_B: (N*T, 3)
        labels: (N*T,) 0 = comp A, 1 = comp B
        positions: (N*T,) context position t
        posteriors: (N*T,) P(comp A | x[0:t])
    """
    model.eval()
    seqs, seq_labels = mixture.sample_batch(n_seqs, seq_len)
    x_in = torch.tensor(seqs[:, :-1], dtype = torch.long, device = device)

    with torch.no_grad():
        _, residuals = model(x_in, return_residuals = True)
        # residuals: list[n_layers + 1] of (N, T, D) cpu tensors
        all_acts = [r.numpy() for r in residuals]  # keep as (N, T, D)

    N, T, D = all_acts[0].shape

    beliefs_A = []
    beliefs_B = []
    labels_out = []
    positions = []
    posteriors = []
    log_prior_A = np.log(mixture.mixing_prob)
    log_prior_B = np.log(1 - mixture.mixing_prob)

    for i, seq in enumerate(seqs):
        log_pA = log_prior_A
        log_pB = log_prior_B
        eta_A = mixture.comps[0].pi.copy()
        eta_B = mixture.comps[1].pi.copy()

        for t in range(T):
            tok = seq[t]
            eta_A_new = eta_A @ mixture.comps[0].T[tok]
            eta_B_new = eta_B @ mixture.comps[1].T[tok]
            p_tok_A = (eta_A * mixture.comps[0].T[tok].sum(axis = 1)).sum()
            p_tok_B = (eta_B * mixture.comps[1].T[tok].sum(axis = 1)).sum()
            log_pA += np.log(p_tok_A + 1e-300)
            log_pB += np.log(p_tok_B + 1e-300)
            eta_A = eta_A_new / eta_A_new.sum()
            eta_B = eta_B_new / eta_B_new.sum()
            post_A = np.exp(log_pA - np.logaddexp(log_pA, log_pB))

            beliefs_A.append(eta_A.copy())
            beliefs_B.append(eta_B.copy())
            labels_out.append(seq_labels[i])
            positions.append(t)
            posteriors.append(post_A)

    # reshape activations from (N, T, D) -> (N*T, D)
    flat_acts = [a.reshape(N * T, D) for a in all_acts]

    return (
        flat_acts,
        np.array(beliefs_A, dtype = np.float32),
        np.array(beliefs_B, dtype = np.float32),
        np.array(labels_out, dtype = int),
        np.array(positions, dtype = int),
        np.array(posteriors, dtype = np.float32),
    )


# Probe helpers (identical to mess3_analysis_probe.py)
def fit_probe(activations, targets):
    A = np.hstack([activations, np.ones((len(activations), 1))])
    W, *_ = np.linalg.lstsq(A, targets, rcond = None)
    return W


def r2_score(true, pred):
    r2s = []
    for s in range(true.shape[1]):
        ss_res = ((true[:, s] - pred[:, s]) ** 2).sum()
        ss_tot = ((true[:, s] - true[:, s].mean()) ** 2).sum()
        r2s.append(float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0)
    return float(np.mean(r2s))


def apply_probe(acts, W):
    A = np.hstack([acts, np.ones((len(acts), 1))])
    return A @ W


def get_subspace_basis(W, d_model, n_components):
    """
    top-n_components left singular vectors of the weight matrix (bias dropped).
    """
    U, _, _ = np.linalg.svd(W[:d_model], full_matrices = False)
    return U[:, :n_components]


def subspace_overlap(Q1, Q2):
    """
    (1 / d_min) * ||Q1.T @ Q2||_F^2 in [0, 1].
    """
    M = Q1.T @ Q2
    d_min = min(Q1.shape[1], Q2.shape[1])
    return float(np.sum(M ** 2) / d_min)


# R^2 across layers
def plot_r2_across_layers(all_acts, beliefs_A, beliefs_B, posteriors, positions, pos_idx, tag = ""):
    """
    At the chosen context position, fit probes for eta_A, eta_B, post_A at
    every layer and plot mean R^2 vs layer index.

    parameters
        pos_idx : int -- context position to evaluate at (e.g. last position)
    """
    mask = positions == pos_idx
    bA_t = beliefs_A[mask]
    bB_t = beliefs_B[mask]
    post_t = posteriors[mask].reshape(-1, 1)

    n_layers = len(all_acts)  # embedding + n transformer blocks
    r2_A = []
    r2_B = []
    r2_post = []

    for layer_idx, acts in enumerate(all_acts):
        acts_t = acts[mask]
        W_A = fit_probe(acts_t, bA_t)
        W_B = fit_probe(acts_t, bB_t)
        W_post = fit_probe(acts_t, post_t)

        r2_A.append(r2_score(bA_t, apply_probe(acts_t, W_A)))
        r2_B.append(r2_score(bB_t, apply_probe(acts_t, W_B)))
        r2_post.append(r2_score(post_t, apply_probe(acts_t, W_post)))

    layer_labels = ["Emb"] + [f"L{i + 1}" for i in range(n_layers - 1)]
    x = np.arange(n_layers)

    fig, ax = plt.subplots(figsize = (7, 4))

    ax.plot(x, r2_A, "o-", color = "#f77f00", lw = 2, ms = 6, label = "eta_A  (comp A belief)")
    ax.plot(x, r2_B, "s-", color = "#4361ee", lw = 2, ms = 6, label = "eta_B  (comp B belief)")
    ax.plot(x, r2_post, "^-", color = "#2dc653", lw = 2, ms = 6, label = "post_A  (component posterior)")

    ax.axhline(1.0, ls = "--", color = "gray", lw = 1, label = "Perfect R^2 = 1")
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, fontsize = 10)
    ax.set(
        xlabel = "Layer",
        ylabel = "Mean R^2",
        title = f"Belief probe R^2 across depth  (context position t = {pos_idx})",
        ylim = (-0.05, 1.08),
    )
    ax.legend(fontsize = 9)
    ax.grid(axis = "y", alpha = 0.3)
    plt.tight_layout()

    fname = f"r2_across_layers{tag}.png"
    plt.savefig(fname, dpi = 150, bbox_inches = "tight")
    plt.close()

    return r2_A, r2_B, r2_post


# subspace orthogonality across layers
def plot_orthogonality_across_layers(all_acts, beliefs_A, beliefs_B, posteriors, positions, pos_idx, d_model, n_components = 2, n_shuffles = 300, tag = ""):
    """
    At the chosen context position, extract probe subspaces at every layer and compute pairwise overlaps:
    """
    mask = positions == pos_idx
    bA_t = beliefs_A[mask]
    bB_t = beliefs_B[mask]
    post_t = posteriors[mask].reshape(-1, 1)

    n_layers = len(all_acts)
    ov_AB = []
    ov_Ap = []
    ov_Bp = []

    for acts in all_acts:
        acts_t = acts[mask]
        W_A = fit_probe(acts_t, bA_t)
        W_B = fit_probe(acts_t, bB_t)
        W_post = fit_probe(acts_t, post_t)

        Q_A = get_subspace_basis(W_A, d_model, n_components)
        Q_B = get_subspace_basis(W_B, d_model, n_components)
        Q_post = get_subspace_basis(W_post, d_model, 1)

        ov_AB.append(subspace_overlap(Q_A, Q_B))
        ov_Ap.append(subspace_overlap(Q_A, Q_post))
        ov_Bp.append(subspace_overlap(Q_B, Q_post))

    # null distribution at the last layer (permuted probes)
    rng = np.random.default_rng(42)
    acts_last = all_acts[-1][mask]
    null_AB = []
    null_Ap = []
    null_Bp = []

    for _ in range(n_shuffles):
        perm = rng.permutation(len(bA_t))
        Q_A_r = get_subspace_basis(fit_probe(acts_last, bA_t[perm]), d_model, n_components)
        Q_B_r = get_subspace_basis(fit_probe(acts_last, bB_t[perm]), d_model, n_components)
        Q_post_r = get_subspace_basis(fit_probe(acts_last, post_t[perm]), d_model, 1)
        null_AB.append(subspace_overlap(Q_A_r, Q_B_r))
        null_Ap.append(subspace_overlap(Q_A_r, Q_post_r))
        null_Bp.append(subspace_overlap(Q_B_r, Q_post_r))

    null_means = [np.mean(null_AB), np.mean(null_Ap), np.mean(null_Bp)]
    null_stds = [np.std(null_AB), np.std(null_Ap), np.std(null_Bp)]

    layer_labels = ["Emb"] + [f"L{i + 1}" for i in range(n_layers - 1)]
    x = np.arange(n_layers)

    fig, ax = plt.subplots(figsize = (7, 4))

    ax.plot(x, ov_AB, "o-", color = "#e63946", lw = 2, ms = 6, label = "overlap(eta_A, eta_B)")
    ax.plot(x, ov_Ap, "s--", color = "#f77f00", lw = 1.8, ms = 5, label = "overlap(eta_A, post_A)")
    ax.plot(x, ov_Bp, "^--", color = "#4361ee", lw = 1.8, ms = 5, label = "overlap(eta_B, post_A)")

    # null band for overlap(eta_A, eta_B) (most important pair)
    ax.axhline(null_means[0], ls = ":", color = "#e63946", lw = 1.2, alpha = 0.6)
    ax.fill_between(
        x,
        null_means[0] - null_stds[0],
        null_means[0] + null_stds[0],
        color = "#e63946",
        alpha = 0.12,
        label = "Null +- sigma  [overlap(eta_A,eta_B)]",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, fontsize = 10)
    ax.set(
        xlabel = "Layer",
        ylabel = "Subspace overlap  (0 = orthogonal, 1 = identical)",
        title = f"Subspace orthogonality across depth  (context position t = {pos_idx})",
        ylim = (-0.02, max(max(ov_AB), null_means[0] + 2 * null_stds[0]) * 1.25),
    )
    ax.legend(fontsize = 8, loc = "upper right")
    ax.grid(axis = "y", alpha = 0.3)
    plt.tight_layout()

    fname = f"orthogonality_across_layers{tag}.png"
    plt.savefig(fname, dpi = 150, bbox_inches = "tight")
    plt.close()

    return ov_AB, ov_Ap, ov_Bp



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type = str, default = "mess3_ckpt.pt")
    parser.add_argument("--n_seqs", type = int, default = 3000)
    parser.add_argument(
        "--pos",
        type = str,
        default = "last",
        help = 'Context position to evaluate at. "last" uses the final position; an integer uses that specific t.',
    )
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

    all_acts, bA, bB, labels, positions, posteriors = collect_all_layers(
        model,
        mixture,
        n_seqs = args.n_seqs,
        seq_len = seq_len,
        device = device,
    )

    # resolve position index
    T_vals = np.unique(positions)
    pos_idx = T_vals[-1] if args.pos == "last" else int(args.pos)
    if pos_idx not in T_vals:
        raise ValueError(f"--pos {pos_idx} not in available positions {T_vals}")

    tag = f"_pos{pos_idx}"

    r2_A, r2_B, r2_post = plot_r2_across_layers(
        all_acts,
        bA,
        bB,
        posteriors,
        positions,
        pos_idx = pos_idx,
        tag = tag,
    )

    ov_AB, ov_Ap, ov_Bp = plot_orthogonality_across_layers(
        all_acts,
        bA,
        bB,
        posteriors,
        positions,
        pos_idx = pos_idx,
        d_model = d_model,
        n_components = 2,
        n_shuffles = 300,
        tag = tag,
    )

    return {
        "r2_A": r2_A,
        "r2_B": r2_B,
        "r2_post": r2_post,
        "ov_AB": ov_AB,
        "ov_Ap": ov_Ap,
        "ov_Bp": ov_Bp,
        "labels": labels,
    }


if __name__ == "__main__":
    results = main()