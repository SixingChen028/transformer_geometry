import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Experiment config
SEQ_LEN = 16
VOCAB_SIZE = 3
D_MODEL = 120
N_HEADS = 3
N_LAYERS = 4


# Mess3 processes
class Mess3:
    """
    3-state HMM.
    """

    def __init__(self, alpha = 0.6, x = 0.15):
        self.alpha = alpha
        self.x = x
        b = (1 - alpha) / 2
        y = 1 - 2 * x
        self.T = np.array([
            [[alpha * y, b * x, b * x],
             [alpha * x, b * y, b * x],
             [alpha * x, b * x, b * y]],
            [[b * y, alpha * x, b * x],
             [b * x, alpha * y, b * x],
             [b * x, alpha * x, b * y]],
            [[b * y, b * x, alpha * x],
             [b * x, b * y, alpha * x],
             [b * x, b * x, alpha * y]],
        ])  # (3 tokens, 3 from-states, 3 to-states)
        self.pi = np.ones(3) / 3

    def sample_sequences(self, n_seqs, seq_len):
        """
        return (n_seqs, seq_len) int64 token array.
        """
        seqs = np.empty((n_seqs, seq_len), dtype = np.int64)
        state = np.random.choice(3, size = n_seqs, p = self.pi)
        for t in range(seq_len):
            p_tok = self.T[:, state, :].sum(axis = 2).T  # (n_seqs, 3)
            cdf = p_tok.cumsum(axis = 1)
            token = (np.random.rand(n_seqs, 1) > cdf).sum(axis = 1).astype(np.int64)
            seqs[:, t] = token

            p_next = self.T[token, state, :]
            p_next /= p_next.sum(axis = 1, keepdims = True)
            state = (np.random.rand(n_seqs, 1) > p_next.cumsum(axis = 1)).sum(axis = 1)
        return seqs

    def predictive_vector(self, context):
        """
        bayesian belief state eta = P(hidden | x[0:t]).
        returns (3,) array.
        """
        eta = self.pi.copy()
        for tok in context:
            eta = eta @ self.T[tok]
            eta /= eta.sum()
        return eta

    def token_probs(self, eta):
        """
        P(next token | belief state eta).
        returns (3,) array.
        """
        return np.array([(eta @ self.T[tok]).sum() for tok in range(3)])


class Mess3Mixture:
    """
    non-ergodic mixture: each sequence is generated entirely by one of two Mess3 components, chosen once per sequence.
    """

    def __init__(self, comp_A: Mess3, comp_B: Mess3, mixing_prob = 0.5):
        self.comps = [comp_A, comp_B]
        self.mixing_prob = mixing_prob  # P(component A)

    def sample_batch(self, n_seqs, seq_len):
        """
        returns
            seqs : (n_seqs, seq_len) int64
            labels : (n_seqs,) int, 0 = comp_A, 1 = comp_B
        """
        labels = (np.random.rand(n_seqs) > self.mixing_prob).astype(int)
        seqs = np.empty((n_seqs, seq_len), dtype = np.int64)
        for c in [0, 1]:
            idx = np.where(labels == c)[0]
            if len(idx):
                seqs[idx] = self.comps[c].sample_sequences(len(idx), seq_len)
        return seqs, labels


def make_components():
    """
    canonical component parameters for this experiment.
    """
    return Mess3(alpha = 0.95, x = 0.1), Mess3(alpha = 0.05, x = 0.1)


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
    def __init__(
        self,
        vocab_size = VOCAB_SIZE,
        d_model = D_MODEL,
        n_heads = N_HEADS,
        n_layers = N_LAYERS,
        max_seq_len = SEQ_LEN,
    ):
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
        h = self.tok_emb(x) + self.pos_emb(torch.arange(T, device = x.device))
        residuals = [h.detach().cpu()] if return_residuals else None

        for block in self.blocks:
            h = block(h)
            if return_residuals:
                residuals.append(h.detach().cpu())

        logits = self.head(self.ln_f(h))
        return (logits, residuals) if return_residuals else logits


# Analysis helpers
def get_residuals_and_beliefs(model, mixture, n_seqs = 2000, seq_len = SEQ_LEN, layer_idx = -1, device = "cpu"):
    """
    sample sequences, run forward pass, collect residual stream activations and bayesian quantities at every context position.

    returns (all flat over N*T positions)
        acts : (N*T, D) residual stream at the requested layer
        beliefs_A : (N*T, 3) belief state under comp A given context so far
        beliefs_B : (N*T, 3) belief state under comp B given context so far
        labels : (N*T,) true generating component (0 = A, 1 = B)
        positions : (N*T,) context position index t
        posteriors : (N*T,) P(comp A | x[0..t]), bayesian posterior
    """
    model.eval()
    seqs, seq_labels = mixture.sample_batch(n_seqs, seq_len)
    x_in = torch.tensor(seqs[:, :-1], dtype = torch.long, device = device)

    with torch.no_grad():
        _, residuals = model(x_in, return_residuals = True)
        acts = residuals[layer_idx].numpy()  # (N, T, D)

    N, T, D = acts.shape
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

    return (
        acts.reshape(N * T, D),
        np.array(beliefs_A, dtype = np.float32),
        np.array(beliefs_B, dtype = np.float32),
        np.array(labels_out, dtype = int),
        np.array(positions, dtype = int),
        np.array(posteriors, dtype = np.float32),
    )


def center_activations(activations, labels, positions, mode = "none"):
    """
    mode = 'none' : no centering (preserves full geometry)
    mode = 'global_pos' : subtract per-position mean (removes component signal)
    mode = 'comp_pos' : subtract per-(component, position) mean
    (isolates within-component belief geometry)
    """
    acts = activations.copy()
    if mode == "global_pos":
        for t in np.unique(positions):
            m = positions == t
            acts[m] -= acts[m].mean(axis = 0)
    elif mode == "comp_pos":
        for c in [0, 1]:
            for t in np.unique(positions):
                m = (labels == c) & (positions == t)
                if m.sum():
                    acts[m] -= acts[m].mean(axis = 0)
    return acts


def linear_r2(activations, targets):
    """
    fit OLS and return per-dimension R^2 and predictions.
    """
    A = np.hstack([activations, np.ones((len(activations), 1))])
    W, *_ = np.linalg.lstsq(A, targets, rcond = None)
    pred = A @ W
    r2s = []
    for s in range(targets.shape[1]):
        ss_res = ((targets[:, s] - pred[:, s]) ** 2).sum()
        ss_tot = ((targets[:, s] - targets[:, s].mean()) ** 2).sum()
        r2s.append(float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0)
    return r2s, pred