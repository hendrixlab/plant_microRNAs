# train_4cut_seq_struct_v8_lr_sweep.py
# - Trains the same model with multiple learning rates (LR sweep)
# - Saves ALL plots as .png (no plt.show)
# - Plots:
#     1) For each LR: VAL curves per epoch (loss, mean exact, mean ±1, mean MAE)
#     2) LR comparison: best mean VAL MAE (and best mean VAL exact) vs LR
#     3) TEST: Pred vs True scatter (all 4 cuts in one plot)
#     4) TEST: Signed shift-counts (pred-true) overall in one plot
#
# Input files (3-line blocks):
#   name p1 p2 p3 p4
#   SEQ
#   STRUCT

import os
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")  # IMPORTANT: save PNGs without display
import matplotlib.pyplot as plt

# -------------------------
# Config
# -------------------------
TRAIN_FILE = "train_cutsite_with_seq_struct.txt"
VAL_FILE   = "val_cutsite_with_seq_struct.txt"
TEST_FILE  = "test_cutsite_with_seq_struct.txt"

LABELS = ["5pDrosha", "5pDicer", "3pDicer", "3pDrosha"]

OUTDIR = "plots_v8_lr_sweep"
os.makedirs(OUTDIR, exist_ok=True)

# Sweep these
LR_LIST = [1e-4, 2e-4, 5e-4]

EPOCHS = 50
BATCH_TRAIN = 32
BATCH_EVAL  = 64

SEED = 7

# -------------------------
# Encoding
# -------------------------
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

# -------------------------
# Utils
# -------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def mean_of_list(x4):
    return (x4[0] + x4[1] + x4[2] + x4[3]) / 4.0

def sigma_for_epoch(ep):
    if ep <= 10:
        return 2.0
    if ep <= 25:
        return 1.5
    return 1.0

# -------------------------
# Data
# -------------------------
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

    # Expect multiples of 3, but don't crash if not perfect
    nblocks = len(lines) // 3

    for bi in range(nblocks):
        i = bi * 3
        header = lines[i].split()
        if len(header) < 5:
            continue
        name = header[0]
        cuts = [int(header[1]), int(header[2]), int(header[3]), int(header[4])]

        if one_based:
            cuts = [p - 1 for p in cuts]  # convert to 0-based

        seq = lines[i + 1].upper().replace("T", "U")
        st  = lines[i + 2]
        L = min(len(seq), len(st))
        if L <= 0:
            continue

        # clamp cuts into range
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
        # extra safety
        if b >= "a" and b <= "z":
            b = b.upper()
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

# -------------------------
# Model
# -------------------------
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
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        outp, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(outp, batch_first=True)
        logits = self.proj(out)                 # [B,L,4]
        return logits.transpose(1, 2).contiguous()  # [B,4,L]

# -------------------------
# Loss
# -------------------------
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

# -------------------------
# Metrics / Collect
# -------------------------
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

        logits = model(x, lens)            # [B,4,L]
        pred = logits.argmax(dim=-1)       # [B,4]
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
def collect_preds_all(model, loader, device):
    """
    Returns dict j -> (true_list, pred_list) for j in 0..3
    """
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

# -------------------------
# Plots (PNG)
# -------------------------
def save_val_curves_png(hist, lr, outpath):
    """
    hist keys:
      'train_loss', 'val_exact_mean', 'val_tol1_mean', 'val_mae_mean'
    """
    epochs = list(range(1, len(hist["train_loss"]) + 1))

    plt.figure(figsize=(12, 8))

    # 1) Train loss
    plt.plot(epochs, hist["train_loss"], label="Train loss")

    # 2) Val mean MAE
    plt.plot(epochs, hist["val_mae_mean"], label="VAL mean MAE")

    # 3) Val mean exact + tol1
    plt.plot(epochs, hist["val_exact_mean"], label="VAL mean exact")
    plt.plot(epochs, hist["val_tol1_mean"], label="VAL mean ±1")

    plt.xlabel("Epoch")
    plt.title(f"VAL curves (LR={lr})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def save_lr_comparison_png(results, outpath):
    """
    results: list of dicts with keys:
      lr, best_val_mae_mean, best_val_exact_mean
    """
    lrs = [r["lr"] for r in results]
    best_mae = [r["best_val_mae_mean"] for r in results]
    best_exact = [r["best_val_exact_mean"] for r in results]

    plt.figure(figsize=(12, 5))
    plt.plot(lrs, best_mae, marker="o", label="Best mean VAL MAE (lower better)")
    plt.plot(lrs, best_exact, marker="o", label="Best mean VAL exact (higher better)")
    plt.xscale("log")
    plt.xlabel("Learning rate (log scale)")
    plt.title("LR sweep comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def save_test_pred_vs_true_png(test_data, labels, outpath, title="TEST: Pred vs True (all cuts)"):
    plt.figure(figsize=(7, 7))

    all_true = []
    all_pred = []

    for j in range(4):
        yt, yp = test_data[j]
        if len(yt) == 0:
            continue
        plt.scatter(yt, yp, s=8, alpha=0.6, label=labels[j])
        all_true.extend(yt)
        all_pred.extend(yp)

    if len(all_true) == 0:
        return

    mn = min(min(all_true), min(all_pred))
    mx = max(max(all_true), max(all_pred))
    plt.plot([mn, mx], [mn, mx], linewidth=1)

    plt.xlabel("True position (0-based)")
    plt.ylabel("Predicted position (0-based)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def save_signed_shift_counts_png(test_data, outpath, title="TEST: Signed shift counts (pred - true)", max_shift_show=20):
    # Collect all errors across all 4 cuts
    all_err = []
    for j in range(4):
        yt, yp = test_data[j]
        for t, p in zip(yt, yp):
            all_err.append(int(p) - int(t))

    if len(all_err) == 0:
        return

    # Count shifts
    counts = {}
    for e in all_err:
        if e not in counts:
            counts[e] = 1
        else:
            counts[e] += 1

    exact = counts[0] if 0 in counts else 0
    pm1 = 0
    if -1 in counts:
        pm1 += counts[-1]
    if 1 in counts:
        pm1 += counts[1]

    pm2 = pm1
    if -2 in counts:
        pm2 += counts[-2]
    if 2 in counts:
        pm2 += counts[2]

    total = len(all_err)

    xs = list(range(-max_shift_show, max_shift_show + 1))
    ys = []
    for x in xs:
        ys.append(counts[x] if x in counts else 0)

    left_tail = 0
    right_tail = 0
    for k in counts:
        if k < -max_shift_show:
            left_tail += counts[k]
        elif k > max_shift_show:
            right_tail += counts[k]

    plt.figure(figsize=(14, 5))
    plt.bar(xs, ys)

    # tails (optional)
    if left_tail > 0:
        plt.bar(-max_shift_show - 2, left_tail)
        plt.text(-max_shift_show - 2, left_tail, f"<{-max_shift_show}", ha="center", va="bottom", fontsize=9)
    if right_tail > 0:
        plt.bar(max_shift_show + 2, right_tail)
        plt.text(max_shift_show + 2, right_tail, f">{max_shift_show}", ha="center", va="bottom", fontsize=9)

    plt.axvline(0, linewidth=1)

    # Annotate summary on plot
    # exact rate, ±1 rate, ±2 rate
    exact_rate = exact / total
    pm1_rate = (exact + pm1) / total
    pm2_rate = (exact + pm2) / total
    txt = f"Total={total} | Exact(0)={exact} ({exact_rate:.3f}) | ±1={exact+pm1} ({pm1_rate:.3f}) | ±2={exact+pm2} ({pm2_rate:.3f})"
    plt.title(title)
    plt.xlabel("Signed shift = (pred - true)")
    plt.ylabel("Count")
    plt.text(0.01, 0.98, txt, transform=plt.gca().transAxes, ha="left", va="top", fontsize=9)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# -------------------------
# Train one LR
# -------------------------
def train_for_lr(lr, train_loader, val_loader, device):
    model = ConvBiLSTMCut4(fin=7, hidden=256, layers=2, dropout=0.2, conv_channels=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    hist = {
        "train_loss": [],
        "val_exact_mean": [],
        "val_tol1_mean": [],
        "val_mae_mean": [],
    }

    best_val_mae = 1e18
    best_val_exact = -1.0
    best_path = os.path.join(OUTDIR, f"best_lr_{lr:.0e}.pt")

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

        val_exact, val_tol1, _, val_mae = eval_metrics(model, val_loader, device)
        val_mae_mean = mean_of_list(val_mae)
        val_exact_mean = mean_of_list(val_exact)
        val_tol1_mean = mean_of_list(val_tol1)

        hist["train_loss"].append(train_loss)
        hist["val_exact_mean"].append(val_exact_mean)
        hist["val_tol1_mean"].append(val_tol1_mean)
        hist["val_mae_mean"].append(val_mae_mean)

        print(
            f"[LR={lr:.0e}] Epoch {ep:02d}  loss={train_loss:.4f}  sigma={sigma:.1f}  "
            f"VAL mean MAE={val_mae_mean:.4f}  mean exact={val_exact_mean:.4f}  mean ±1={val_tol1_mean:.4f}"
        )

        # best by mean VAL MAE (primary)
        if val_mae_mean < best_val_mae:
            best_val_mae = val_mae_mean
            best_val_exact = val_exact_mean
            torch.save(model.state_dict(), best_path)

    return {
        "lr": lr,
        "best_path": best_path,
        "hist": hist,
        "best_val_mae_mean": best_val_mae,
        "best_val_exact_mean": best_val_exact,
    }

# -------------------------
# Main
# -------------------------
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load data (files are 1-based -> convert to 0-based)
    train_items = read_cutsite_seq_struct(TRAIN_FILE, one_based=True)
    val_items   = read_cutsite_seq_struct(VAL_FILE,   one_based=True)
    test_items  = read_cutsite_seq_struct(TEST_FILE,  one_based=True)

    print("Train:", len(train_items), "Val:", len(val_items), "Test:", len(test_items))

    train_loader = DataLoader(Cut4Dataset(train_items), batch_size=BATCH_TRAIN, shuffle=True,  collate_fn=pad_collate)
    val_loader   = DataLoader(Cut4Dataset(val_items),   batch_size=BATCH_EVAL,  shuffle=False, collate_fn=pad_collate)
    test_loader  = DataLoader(Cut4Dataset(test_items),  batch_size=BATCH_EVAL,  shuffle=False, collate_fn=pad_collate)

    # Train for each LR
    sweep_results = []
    for lr in LR_LIST:
        res = train_for_lr(lr, train_loader, val_loader, device)
        sweep_results.append(res)

        # Save per-LR curves png
        curves_png = os.path.join(OUTDIR, f"val_curves_lr_{lr:.0e}.png")
        save_val_curves_png(res["hist"], lr, curves_png)
        print("Saved:", curves_png)

    # LR comparison png
    comp_png = os.path.join(OUTDIR, "lr_comparison.png")
    save_lr_comparison_png(sweep_results, comp_png)
    print("Saved:", comp_png)

    # Pick best LR by best mean VAL MAE
    best_idx = 0
    for i in range(1, len(sweep_results)):
        if sweep_results[i]["best_val_mae_mean"] < sweep_results[best_idx]["best_val_mae_mean"]:
            best_idx = i
    best = sweep_results[best_idx]
    best_lr = best["lr"]
    best_path = best["best_path"]

    print("\nBest LR:", f"{best_lr:.0e}", "Best mean VAL MAE:", best["best_val_mae_mean"])

    # Load best model for TEST plots
    model = ConvBiLSTMCut4(fin=7, hidden=256, layers=2, dropout=0.2, conv_channels=128).to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))

    # TEST metrics
    test_exact, test_tol1, test_tol2, test_mae = eval_metrics(model, test_loader, device)
    print("\nTEST (best LR)")
    print(" exact:", test_exact, "mean:", mean_of_list(test_exact))
    print(" ±1  :", test_tol1,  "mean:", mean_of_list(test_tol1))
    print(" ±2  :", test_tol2,  "mean:", mean_of_list(test_tol2))
    print(" MAE :", test_mae,   "mean:", mean_of_list(test_mae))

    # Collect predictions for plots
    test_data = collect_preds_all(model, test_loader, device)

    # Save TEST pred vs true png
    pvst_png = os.path.join(OUTDIR, f"test_pred_vs_true_best_lr_{best_lr:.0e}.png")
    save_test_pred_vs_true_png(test_data, LABELS, pvst_png, title=f"TEST: Pred vs True (LR={best_lr:.0e})")
    print("Saved:", pvst_png)

    # Save TEST signed shift counts png (overall)
    shift_png = os.path.join(OUTDIR, f"test_signed_shift_counts_best_lr_{best_lr:.0e}.png")
    save_signed_shift_counts_png(test_data, shift_png, title=f"TEST: Signed shift counts (LR={best_lr:.0e})", max_shift_show=20)
    print("Saved:", shift_png)

    # Also save final weights
    final_path = os.path.join(OUTDIR, f"cutsite_conv_bilstm_best_lr_{best_lr:.0e}.pt")
    torch.save(model.state_dict(), final_path)
    print("Saved model:", final_path)

    print("\nDone. All PNGs are in:", OUTDIR)

if __name__ == "__main__":
    main()
