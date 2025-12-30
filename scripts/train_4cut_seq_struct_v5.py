# train_4cut_seq_struct_v7.py
# Fix: 1-based labels -> 0-based
# Adds: sigma annealing + best checkpoint saving by VAL MAE
# Uses ONLY split files with 3-line blocks.

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

TRAIN_FILE = "train_cutsite_with_seq_struct.txt"
VAL_FILE   = "val_cutsite_with_seq_struct.txt"
TEST_FILE  = "test_cutsite_with_seq_struct.txt"

LABELS = ["5pDrosha", "5pDicer", "3pDicer", "3pDrosha"]

IUPAC_TO_VEC = {
    "A": (1.0, 0.0, 0.0, 0.0),
    "C": (0.0, 1.0, 0.0, 0.0),
    "G": (0.0, 0.0, 1.0, 0.0),
    "U": (0.0, 0.0, 0.0, 1.0),
    "T": (0.0, 0.0, 0.0, 1.0),
    "R": (0.5, 0.0, 0.5, 0.0),
    "Y": (0.0, 0.5, 0.0, 0.5),
    "S": (0.0, 0.5, 0.5, 0.0),
    "W": (0.5, 0.0, 0.0, 0.5),
    "K": (0.0, 0.0, 0.5, 0.5),
    "M": (0.5, 0.5, 0.0, 0.0),
    "B": (0.0, 1/3, 1/3, 1/3),
    "D": (1/3, 0.0, 1/3, 1/3),
    "H": (1/3, 1/3, 0.0, 1/3),
    "V": (1/3, 1/3, 1/3, 0.0),
    "N": (0.25, 0.25, 0.25, 0.25),
}

PAIRED_OPEN  = set(["(", "[", "{", "<"])
PAIRED_CLOSE = set([")", "]", "}", ">"])


def read_cutsite_seq_struct(path, one_based=True):
    """
    3-line block:
      name p1 p2 p3 p4
      SEQ
      STRUCT
    If one_based=True, converts positions to 0-based by subtracting 1.
    """
    items = []
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    for i in range(0, len(lines), 3):
        header = lines[i].split()
        name = header[0]
        cuts = [int(header[1]), int(header[2]), int(header[3]), int(header[4])]
        if one_based:
            cuts = [p - 1 for p in cuts]  # ✅ FIX

        seq = lines[i + 1].upper().replace("T", "U")
        st  = lines[i + 2]
        L = min(len(seq), len(st))
        if L <= 0:
            continue

        # keep in-range
        cuts2 = []
        for p in cuts:
            if p < 0:
                cuts2.append(0)
            elif p >= L:
                cuts2.append(L - 1)
            else:
                cuts2.append(p)

        items.append((name, seq[:L], st[:L], cuts2))

    return items


def encode_seq_struct(seq, st):
    """
    [L,7]:
      0..3 = A,C,G,U probs
      4    = is_dot
      5    = is_open
      6    = is_close
    """
    L = min(len(seq), len(st))
    x = torch.zeros((L, 7), dtype=torch.float32)
    for i in range(L):
        b = seq[i]
        if b not in IUPAC_TO_VEC:
            b = "N"
        a, c, g, u = IUPAC_TO_VEC[b]
        x[i, 0] = a
        x[i, 1] = c
        x[i, 2] = g
        x[i, 3] = u

        ch = st[i]
        if ch == ".":
            x[i, 4] = 1.0
        elif ch in PAIRED_OPEN:
            x[i, 5] = 1.0
        elif ch in PAIRED_CLOSE:
            x[i, 6] = 1.0
        else:
            x[i, 4] = 1.0
    return x


class Cut4Dataset(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        name, seq, st, cuts = self.items[idx]
        x = encode_seq_struct(seq, st)
        y = torch.tensor(cuts, dtype=torch.long)
        return name, x, y


def pad_collate(batch):
    names = [b[0] for b in batch]
    xs = [b[1] for b in batch]
    ys = torch.stack([b[2] for b in batch], dim=0)
    lens = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)

    Lmax = int(lens.max().item())
    B = len(xs)
    F = xs[0].shape[1]

    xpad = torch.zeros((B, Lmax, F), dtype=torch.float32)
    for i, x in enumerate(xs):
        L = x.shape[0]
        xpad[i, :L] = x

    return names, xpad, ys, lens


class ConvBiLSTMCut4(nn.Module):
    def __init__(self, fin=7, hidden=256, layers=2, dropout=0.2, conv_channels=128):
        super().__init__()
        self.conv = nn.Conv1d(fin, conv_channels, kernel_size=5, padding=2)
        self.act = nn.GELU()
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden * 2, 4)

    def forward(self, x, lens):
        x = x.transpose(1, 2)          # [B,F,L]
        x = self.act(self.conv(x))     # [B,C,L]
        x = x.transpose(1, 2)          # [B,L,C]
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        outp, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(outp, batch_first=True)
        logits = self.proj(out)                 # [B,L,4]
        return logits.transpose(1, 2).contiguous()  # [B,4,L]


def gaussian_targets(y, lens, Lmax, sigma):
    device = y.device
    B = y.shape[0]
    tgt = torch.zeros((B, 4, Lmax), dtype=torch.float32, device=device)
    pos = torch.arange(Lmax, device=device).float()

    for i in range(B):
        Li = int(lens[i].item())
        if Li <= 0:
            continue
        for j in range(4):
            mu = float(y[i, j].item())
            g = torch.exp(-0.5 * ((pos - mu) / sigma) ** 2)
            if Li < Lmax:
                g[Li:] = 0.0
            s = g.sum()
            if s > 0:
                g = g / s
            tgt[i, j, :] = g
    return tgt


def soft_ce_loss(logits, y, lens, sigma):
    logp = nn.functional.log_softmax(logits, dim=-1)
    B, C, Lmax = logp.shape
    tgt = gaussian_targets(y, lens, Lmax, sigma)
    loss = -(tgt * logp).sum(dim=-1)  # [B,4]
    return loss.mean()


@torch.no_grad()
def eval_metrics(model, loader, device):
    model.eval()
    total = 0
    exact = [0, 0, 0, 0]
    tol1  = [0, 0, 0, 0]
    tol2  = [0, 0, 0, 0]
    mae_sum = [0.0, 0.0, 0.0, 0.0]

    for _, x, y, lens in loader:
        x = x.to(device)
        y = y.to(device)
        lens = lens.to(device)

        logits = model(x, lens)
        pred = logits.argmax(dim=-1)
        B = y.shape[0]
        total += B

        for j in range(4):
            d = (pred[:, j] - y[:, j]).abs()
            exact[j] += int((d == 0).sum().item())
            tol1[j]  += int((d <= 1).sum().item())
            tol2[j]  += int((d <= 2).sum().item())
            mae_sum[j] += float(d.float().sum().item())

    if total == 0:
        return [0.0]*4, [0.0]*4, [0.0]*4, [0.0]*4

    exact = [v / total for v in exact]
    tol1  = [v / total for v in tol1]
    tol2  = [v / total for v in tol2]
    mae   = [mae_sum[j] / total for j in range(4)]
    return exact, tol1, tol2, mae


@torch.no_grad()
def collect_test_preds(model, loader, device):
    model.eval()
    data = {0: ([], []), 1: ([], []), 2: ([], []), 3: ([], [])}
    for _, x, y, lens in loader:
        x = x.to(device)
        y = y.to(device)
        lens = lens.to(device)
        logits = model(x, lens)
        pred = logits.argmax(dim=-1)
        for j in range(4):
            data[j][0].extend(y[:, j].cpu().tolist())
            data[j][1].extend(pred[:, j].cpu().tolist())
    return data


def plot_test_only(data, labels):
    for j in range(4):
        yt, yp = data[j]
        if len(yt) == 0:
            continue

        plt.figure()
        plt.scatter(yt, yp, s=6)
        plt.xlabel("True position (0-based)")
        plt.ylabel("Predicted position (0-based)")
        plt.title(f"TEST: True vs Pred ({labels[j]})")
        plt.show()

        err = [p - t for t, p in zip(yt, yp)]
        plt.figure()
        plt.hist(err, bins=60)
        plt.xlabel("Error (pred - true)")
        plt.ylabel("Count")
        plt.title(f"TEST: Error histogram ({labels[j]})")
        plt.show()


def sigma_for_epoch(ep):
    if ep <= 10:
        return 2.0
    if ep <= 25:
        return 1.5
    return 1.0


def mean_of_list(x):
    return (x[0] + x[1] + x[2] + x[3]) / 4.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ one_based=True
    train_items = read_cutsite_seq_struct(TRAIN_FILE, one_based=True)
    val_items   = read_cutsite_seq_struct(VAL_FILE, one_based=True)
    test_items  = read_cutsite_seq_struct(TEST_FILE, one_based=True)

    train_loader = DataLoader(Cut4Dataset(train_items), batch_size=32, shuffle=True,  collate_fn=pad_collate)
    val_loader   = DataLoader(Cut4Dataset(val_items),   batch_size=64, shuffle=False, collate_fn=pad_collate)
    test_loader  = DataLoader(Cut4Dataset(test_items),  batch_size=64, shuffle=False, collate_fn=pad_collate)

    model = ConvBiLSTMCut4(fin=7, hidden=256, layers=2, dropout=0.2, conv_channels=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)

    EPOCHS = 50

    best_val_mae = 1e18
    best_path = "best_cutsite_v7.pt"

    for ep in range(1, EPOCHS + 1):
        model.train()
        sigma = sigma_for_epoch(ep)
        s = 0.0
        n = 0

        for _, x, y, lens in train_loader:
            x = x.to(device)
            y = y.to(device)
            lens = lens.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x, lens)
            loss = soft_ce_loss(logits, y, lens, sigma=sigma)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            s += float(loss.item())
            n += 1

        train_loss = s / max(1, n)

        val_exact, val_tol1, val_tol2, val_mae = eval_metrics(model, val_loader, device)
        val_mae_mean = mean_of_list(val_mae)

        print(
            f"Epoch {ep:02d}  loss={train_loss:.4f}  sigma={sigma:.1f}\n"
            f"  VAL exact={val_exact}\n"
            f"  VAL ±1  ={val_tol1}\n"
            f"  VAL ±2  ={val_tol2}\n"
            f"  VAL MAE ={val_mae}"
        )

        if val_mae_mean < best_val_mae:
            best_val_mae = val_mae_mean
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ saved new best: {best_path} (mean VAL MAE={best_val_mae:.4f})")

    # load best for test
    model.load_state_dict(torch.load(best_path, map_location=device))

    test_exact, test_tol1, test_tol2, test_mae = eval_metrics(model, test_loader, device)
    print("\nTEST (best checkpoint)")
    print(" exact:", test_exact)
    print(" ±1  :", test_tol1)
    print(" ±2  :", test_tol2)
    print(" MAE :", test_mae)

    test_data = collect_test_preds(model, test_loader, device)
    plot_test_only(test_data, LABELS)

    torch.save(model.state_dict(), "cutsite_conv_bilstm_v7_final.pt")
    print("\nSaved: cutsite_conv_bilstm_v7_final.pt")


if __name__ == "__main__":
    main()
