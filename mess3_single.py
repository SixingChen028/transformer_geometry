import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Mess3 process
class Mess3:
    """
    3-state HMM.
    """

    def __init__(self, alpha = 0.6, x = 0.15):
        b = (1 - alpha) / 2
        y = 1 - 2 * x
        self.T = np.array([
            [[alpha * y, b * x, b * x],   # T(token = 0)
             [alpha * x, b * y, b * x],
             [alpha * x, b * x, b * y]],
            [[b * y, alpha * x, b * x],   # T(token = 1)
             [b * x, alpha * y, b * x],
             [b * x, alpha * x, b * y]],
            [[b * y, b * x, alpha * x],   # T(token = 2)
             [b * x, b * y, alpha * x],
             [b * x, b * x, alpha * y]],
        ])  # shape: (3 tokens, 3 from-states, 3 to-states)
        self.pi = np.ones(3) / 3  # uniform stationary distribution

    def sample_batch(self, n_seqs, seq_len):
        """
        sample a fresh batch of sequences. shape: (n_seqs, seq_len).
        """
        seqs = np.empty((n_seqs, seq_len), dtype = np.int64)
        state = np.random.choice(3, size = n_seqs, p = self.pi)
        for t in range(seq_len):
            # P(token | state): vectorised over the batch
            p_tok = self.T[:, state, :].sum(axis = 2).T  # (n_seqs, 3)

            # sample tokens
            cdf = p_tok.cumsum(axis = 1)
            u = np.random.rand(n_seqs, 1)
            token = (u > cdf).sum(axis = 1).astype(np.int64)
            seqs[:, t] = token

            # transition: P(s' | s, token)
            p_next = self.T[token, state, :]  # (n_seqs, 3)
            p_next = p_next / p_next.sum(axis = 1, keepdims = True)
            cdf2 = p_next.cumsum(axis = 1)
            u2 = np.random.rand(n_seqs, 1)
            state = (u2 > cdf2).sum(axis = 1)
        return seqs

    def predictive_vector(self, context):
        """
        bayesian belief: P(hidden_state | observed tokens).
        context: list/array of token indices. empty -> returns stationary pi.
        """
        eta = self.pi.copy()
        for tok in context:
            eta = eta @ self.T[tok]
            eta /= eta.sum()
        return eta


# Transformer
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first = True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device = x.device), diagonal = 1).bool()
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask = mask)[0]
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    architecture: 4 layers, d_model = 120, d_mlp = 480, 3 heads, L = 8.
    """
    def __init__(self, vocab_size = 3, d_model = 120, n_heads = 3, n_layers = 4, max_seq_len = 8):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 4)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias = False)

    def forward(self, x, return_residuals = False):
        B, T = x.shape
        pos = torch.arange(T, device = x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)

        residuals = [h.detach().cpu()] if return_residuals else None
        for block in self.blocks:
            h = block(h)
            if return_residuals:
                residuals.append(h.detach().cpu())

        logits = self.head(self.ln_f(h))
        return (logits, residuals) if return_residuals else logits


# Training
def train(model, mess3, n_steps = 5000, batch_size = 25000, seq_len = 8, lr = 5e-4, device = "cpu"):
    """
    sample a fresh batch of sequences at every gradient step.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 0.0)

    losses = []
    model.train()
    for step in range(1, n_steps + 1):
        # fresh data every step
        seqs = mess3.sample_batch(batch_size, seq_len)
        batch = torch.tensor(seqs, dtype = torch.long, device = device)
        x, y = batch[:, :-1], batch[:, 1:]  # (B, L-1) each

        logits = model(x)  # (B, L-1, vocab)
        loss = F.cross_entropy(logits.reshape(-1, 3), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses


# Geometry analysis
def get_residuals_and_beliefs(model, mess3, n_seqs = 3000, seq_len = 8, layer_idx = -1, device = "cpu"):
    """
    sample fresh analysis sequences, collect residual stream at `layer_idx`,
    and compute paired ground-truth belief vectors.

    alignment: at residual position t (0-indexed), the model has processed
    input tokens x[0..t], so the matching belief is predictive_vector(seq[0 : t + 1])

    layer_idx: 0 = after embedding, 1..n_layers = after that block, -1 = last block.
    """
    model.eval()
    data = mess3.sample_batch(n_seqs, seq_len)  # (N, L)
    x_in = torch.tensor(data[:, :-1], dtype = torch.long, device = device)  # (N, L-1)

    with torch.no_grad():
        _, residuals = model(x_in, return_residuals = True)
        acts = residuals[layer_idx].numpy()  # (N, L-1, d_model)

    N, T, D = acts.shape

    beliefs = []
    for seq in data:
        for t in range(T):
            beliefs.append(mess3.predictive_vector(seq[:t + 1]))

    positions = np.tile(np.arange(T), N)  # (N*T,) context position for each row
    return acts.reshape(N * T, D), np.array(beliefs, dtype = np.float32), positions


def linear_r2(activations, beliefs):
    """
    linear map activations -> beliefs
    returns (R^2 per state, predictions).
    """
    A = np.hstack([activations, np.ones((len(activations), 1))])
    W, *_ = np.linalg.lstsq(A, beliefs, rcond = None)
    pred = A @ W
    r2s = []
    for s in range(beliefs.shape[1]):
        ss_res = ((beliefs[:, s] - pred[:, s]) ** 2).sum()
        ss_tot = ((beliefs[:, s] - beliefs[:, s].mean()) ** 2).sum()
        r2s.append(float(1 - ss_res / ss_tot))
    return r2s, pred


# Plotting
COLORS = ["#e41a1c", "#377eb8", "#4daf4a"]
ENTROPY_RATE = 1.0933  # true entropy rate of Mess3 alpha = 0.6, x = 0.15


def mean_center_per_position(activations, positions):
    """
    mean-center activations within each context position.
    removes positional variance (positional embeddings, position-specific
    attention patterns) to isolate belief geometry
    """
    acts_c = activations.copy()
    for t in np.unique(positions):
        mask = positions == t
        acts_c[mask] -= acts_c[mask].mean(axis = 0)
    return acts_c


def plot_geometry(losses, activations, beliefs, positions, tag = ""):
    # mean-center per position before PCA -> isolates belief geometry
    acts_c = mean_center_per_position(activations, positions)
    pca_a = PCA(n_components = 2).fit(acts_c)
    acts_2d = pca_a.transform(acts_c)

    pca_b = PCA(n_components = 2).fit(beliefs)
    bel_2d = pca_b.transform(beliefs)
    dom = beliefs.argmax(axis = 1)

    r2s, pred = linear_r2(activations, beliefs)
    pred_2d = pca_b.transform(np.clip(pred, 0, 1))

    fig, axes = plt.subplots(1, 3, figsize = (15, 4))
    fig.suptitle(f"Single Mess3 — Residual Stream - Last Block", fontsize = 12)

    # mean-centered residual stream PCA colored by dominant belief state
    for s in range(3):
        m = dom == s
        axes[0].scatter(
            acts_2d[m, 0],
            acts_2d[m, 1],
            c = COLORS[s],
            s = 3,
            alpha = 0.3,
            label = f"State {s}",
        )
    axes[0].set(
        title = "Residual stream PCA",
        xlabel = "PC1",
        ylabel = "PC2",
    )
    axes[0].legend(markerscale = 4, fontsize = 8)

    # ground-truth belief 2-simplex
    for s in range(3):
        m = dom == s
        axes[1].scatter(
            bel_2d[m, 0],
            bel_2d[m, 1],
            c = COLORS[s],
            s = 3,
            alpha = 0.3,
            label = f"State {s}",
        )
    axes[1].set(
        title = "Ground-truth belief vectors",
        xlabel = "PC1",
        ylabel = "PC2",
    )
    axes[1].legend(markerscale = 4, fontsize = 8)

    # linear probe predictions overlaid on true simplex
    axes[2].scatter(
        bel_2d[:, 0],
        bel_2d[:, 1],
        c = "lightgray",
        s = 2,
        alpha = 0.15,
        label = "true beliefs",
    )
    for s in range(3):
        m = dom == s
        axes[2].scatter(pred_2d[m, 0], pred_2d[m, 1], c = COLORS[s], s = 2, alpha = 0.3)
    axes[2].set(
        title = f"Linear probe (mean R^2 = {np.mean(r2s):.3f})",
        xlabel = "PC1 (belief space)",
        ylabel = "PC2",
    )

    plt.tight_layout()
    fname = f"mess3_geometry{tag}.png"
    plt.savefig(fname, dpi = 150, bbox_inches = "tight")
    plt.show()
    return r2s


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # mess3
    mess3 = Mess3(alpha = 0.05, x = 0.1)  # (0.6, 0.15)

    # model
    # 4 layers, d_model = 120, d_mlp = 480, 3 heads, seq_len = 8
    L = 8
    model = GPT(vocab_size = 3, d_model = 120, n_heads = 3, n_layers = 4, max_seq_len = L)

    # train
    # batch_size = 25000, Adam lr = 5e-4, no weight decay, 5000 steps, fresh data each step
    losses = train(
        model,
        mess3,
        n_steps = 5000,
        batch_size = 25000,
        seq_len = L,
        lr = 5e-4,
        device = device,
    )

    # analysis layer by layer
    layer_names = ["embedding"] + [f"block{i + 1}" for i in range(4)]
    layer_results = {}
    for li, lname in enumerate(layer_names):
        acts, beliefs, positions = get_residuals_and_beliefs(
            model,
            mess3,
            n_seqs = 3000,
            seq_len = L,
            layer_idx = li,
            device = device,
        )
        r2s, _ = linear_r2(acts, beliefs)
        layer_results[lname] = {
            "r2s": r2s,
            "mean_r2": float(np.mean(r2s)),
        }

    # full geometry plot for last block
    acts, beliefs, positions = get_residuals_and_beliefs(
        model,
        mess3,
        n_seqs = 3000,
        seq_len = L,
        layer_idx = -1,
        device = device,
    )
    final_r2s = plot_geometry(losses, acts, beliefs, positions, tag = "_lastblock")

    return {
        "losses": losses,
        # "layer_results": layer_results,
        "final_r2s": final_r2s,
    }


if __name__ == "__main__":
    results = main()