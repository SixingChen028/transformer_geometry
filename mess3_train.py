import argparse
import numpy as np
import torch
import torch.nn.functional as F

from mess3_common import (
    Mess3Mixture,
    GPT,
    make_components,
    SEQ_LEN,
    VOCAB_SIZE,
    D_MODEL,
    N_HEADS,
    N_LAYERS,
)


def train(model, mixture, n_steps = 5000, batch_size = 5000, seq_len = SEQ_LEN, lr = 5e-4, device = "cpu", log_every = 500):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 0.0)
    losses = []

    model.train()
    for step in range(1, n_steps + 1):
        seqs, _ = mixture.sample_batch(batch_size, seq_len)
        x = torch.tensor(seqs[:, :-1], dtype = torch.long, device = device)
        y = torch.tensor(seqs[:, 1:], dtype = torch.long, device = device)
        loss = F.cross_entropy(model(x).reshape(-1, VOCAB_SIZE), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step % log_every == 0:
            print(f"step {step}/{n_steps} loss = {np.mean(losses[-log_every:]):.4f}")

    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type = int, default = 5000)
    parser.add_argument("--batch", type = int, default = 25000)
    parser.add_argument("--lr", type = float, default = 5e-4)
    parser.add_argument("--out", type = str, default = "mess3_ckpt.pt")
    parser.add_argument("--seed", type = int, default = 42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    comp_A, comp_B = make_components()
    mixture = Mess3Mixture(comp_A, comp_B, mixing_prob = 0.5)

    model = GPT(
        vocab_size = VOCAB_SIZE,
        d_model = D_MODEL,
        n_heads = N_HEADS,
        n_layers = N_LAYERS,
        max_seq_len = SEQ_LEN,
    )

    losses = train(
        model,
        mixture,
        n_steps = args.steps,
        batch_size = args.batch,
        seq_len = SEQ_LEN,
        lr = args.lr,
        device = device,
    )

    torch.save(
        {
            "model_state": model.state_dict(),
            "losses": losses,
            # store hparams so the analysis script can reconstruct exactly
            "comp_A_alpha": comp_A.alpha,
            "comp_A_x": comp_A.x,
            "comp_B_alpha": comp_B.alpha,
            "comp_B_x": comp_B.x,
            "seq_len": SEQ_LEN,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "vocab_size": VOCAB_SIZE,
            "seed": args.seed,
        },
        args.out,
    )


if __name__ == "__main__":
    main()