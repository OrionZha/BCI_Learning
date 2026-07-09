#!/usr/bin/env python3
"""
EEGNet for BCI Competition IV 2a — 4-class Motor Imagery Classification.
Grid search: epochs × batch_size → test loss & accuracy curves.

Architecture:
  Block 1: Conv2d(1×64) → DepthwiseConv2d(C×1) → BN → ELU → AvgPool(1×4) → Dropout(0.25)
  Block 2: SeparableConv2d(1×16) → BN → ELU → AvgPool(1×8) → Dropout(0.25)
  Classifier: Flatten → Dense → LogSoftmax

Usage:
  python eegnet_bci.py
"""

import os
import time
import warnings
from itertools import product

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ========================= 可调参数 =========================

EPOCHS_LIST = [1500]                        # 测试的 epoch 值
BATCH_LIST = [100, 130, 150, 180, 200]      # 测试的 batch 值
LEARNING_RATE = 0.001
SUBJECT = 1                                 # 被试编号 (1–9)
TMIN = 0.5                                  # MI 窗口起始 (相对 cue)
TMAX = 2.5                                  # MI 窗口结束 (相对 cue)
OUTPUT_DIR = "eegnet_result"
CACHE_FILE = os.path.join(OUTPUT_DIR, "cache.npz")  # 断点续跑缓存

# ============================================================

# BCIC IV 2a channel order (22 EEG + 3 EOG)
CLASS_NAMES = ['left_hand', 'right_hand', 'feet', 'tongue']

# 预处理参数
SFREQ = 250
LOWCUT = 2.0
HIGHCUT = 40.0
BASELINE_SEC = 0.5


# ========================== EEGNet ==========================

class EEGNet(nn.Module):
    def __init__(self, n_channels, n_samples, n_classes,
                 F1=8, D=2, F2=16, dropout=0.25):
        super(EEGNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding='same'),
            nn.Conv2d(F1, D * F1, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(D * F1, D * F1, (1, 16), groups=D * F1,
                      padding='same', bias=False),
            nn.Conv2d(D * F1, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        self.feat_dim = F2 * (n_samples // 32)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feat_dim, n_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


# ========================== Data ==========================

def _bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)


def load_bcic_iv_2a(subject=1, tmin=0.5, tmax=2.5, data_dir=None):
    import glob

    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(__file__), "mne_data", "MNE-bnci-data",
            "~bci", "database", "001-2014",
        )

    sfreq = SFREQ
    cue_sample = int(2.0 * sfreq)
    mi_start = cue_sample + int(tmin * sfreq)
    mi_end = cue_sample + int(tmax * sfreq)
    n_samples = mi_end - mi_start
    base_start = cue_sample - int(BASELINE_SEC * sfreq)
    base_end = cue_sample
    n_baseline = base_end - base_start

    print(f"Loading BCI Competition IV 2a — subject {subject}")
    print(f"  Preprocessing: bandpass {LOWCUT}–{HIGHCUT} Hz | "
          f"baseline [{BASELINE_SEC}s pre-cue] | per-trial per-channel z-score")
    print(f"  MI window: [{tmin}, {tmax}]s post-cue → {n_samples} samples")

    def extract_epochs(mat_path):
        data = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
        run_array = data["data"] if isinstance(data["data"], np.ndarray) else [data["data"]]
        X_list, y_list = [], []
        for run in run_array:
            if len(run.trial) == 0:
                continue
            trials = run.trial.ravel().astype(int)
            labels = run.y.ravel().astype(int)
            eeg = run.X.astype(np.float64)
            eeg_filt = _bandpass_filter(eeg, LOWCUT, HIGHCUT, sfreq)
            for trial_pos, label in zip(trials, labels):
                t0 = trial_pos - 1
                bl = eeg_filt[t0 + base_start:t0 + base_end, :22]
                ep = eeg_filt[t0 + mi_start:t0 + mi_end, :22]
                if ep.shape[0] < n_samples or bl.shape[0] < n_baseline:
                    continue
                epoch = ep - bl.mean(axis=0, keepdims=True)
                ch_std = epoch.std(axis=0, keepdims=True) + 1e-8
                epoch = (epoch - epoch.mean(axis=0, keepdims=True)) / ch_std
                X_list.append(epoch.T)
                y_list.append(label - 1)
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

    train_pattern = os.path.join(data_dir, f"A{subject:02d}T.mat")
    test_pattern = os.path.join(data_dir, f"A{subject:02d}E.mat")
    train_paths = glob.glob(train_pattern)
    test_paths = glob.glob(test_pattern)

    if not train_paths:
        raise FileNotFoundError(f"No training file: {train_pattern}")

    print(f"  loading {os.path.basename(train_paths[0])} …")
    X_train, y_train = extract_epochs(train_paths[0])

    if test_paths:
        print(f"  loading {os.path.basename(test_paths[0])} …")
        X_test, y_test = extract_epochs(test_paths[0])
    else:
        print(f"  A{subject:02d}E.mat not found — using 20% hold-out")
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train,
        )

    n_ch = X_train.shape[1]
    print(f"  train: {X_train.shape}  test: {X_test.shape}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name:>12s}: train={sum(y_train==i):3d}  test={sum(y_test==i):3d}")

    X_train = X_train[:, np.newaxis, :, :]
    X_test = X_test[:, np.newaxis, :, :]

    return (X_train, y_train), (X_test, y_test), {
        "n_channels": n_ch, "n_samples": n_samples, "n_classes": 4,
    }


# ======================= Training =======================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        correct += output.argmax(1).eq(target).sum().item()
        total += data.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        total_loss += F.nll_loss(output, target, reduction='sum').item()
        correct += output.argmax(1).eq(target).sum().item()
        total += data.size(0)
    return total_loss / total, correct / total


def build_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def run_one(epochs, batch_size, X_train, y_train, X_test, y_test,
            n_ch, n_times, n_cls, device):
    """Train one model; return (test_losses, test_accs) lists of length `epochs`."""
    train_set = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train),
    )
    test_set = TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test),
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = EEGNet(n_ch, n_times, n_cls).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    test_losses, test_accs = [], []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_epoch(model, train_loader, optimizer, device)
        t_loss, t_acc = eval_epoch(model, test_loader, device)
        scheduler.step()
        test_losses.append(t_loss)
        test_accs.append(t_acc)

        if epoch % 100 == 0 or epoch == epochs:
            elapsed = time.time() - t0
            print(f"    epoch {epoch:4d}/{epochs}  "
                  f"test_loss={t_loss:.4f}  test_acc={t_acc:.2%}  "
                  f"[{elapsed:.0f}s]")

    return test_losses, test_accs


# ======================= Plotting =======================

# 颜色和线型：动态分配，支持任意 batch 值
BATCH_COLORS = plt.cm.tab10.colors  # 最多 10 种颜色


def _get_color(batch_idx):
    return BATCH_COLORS[batch_idx % len(BATCH_COLORS)]

def _get_style(batch_idx):
    return ['-', '--', '-.', ':', (0, (3, 1, 1, 1))][batch_idx % 5]


# ======================= Caching =======================

def save_cache(results, cache_path):
    """保存结果到 npz 文件，key 格式为 '1500_100_loss', '1500_100_acc' 等."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    flat = {}
    for (ep, bs), (losses, accs) in results.items():
        flat[f"{ep}_{bs}_loss"] = np.array(losses, dtype=np.float32)
        flat[f"{ep}_{bs}_acc"] = np.array(accs, dtype=np.float32)
    np.savez(cache_path, **flat)
    print(f"Cache saved → {cache_path}")


def load_cache(cache_path):
    """加载缓存，返回 {(epochs, batch): (losses, accs)} 或空 dict."""
    if not os.path.exists(cache_path):
        return {}
    data = np.load(cache_path)
    results = {}
    for key in data.files:
        if key.endswith('_loss'):
            parts = key.replace('_loss', '').split('_', 1)
            ep, bs = int(parts[0]), int(parts[1])
            losses = data[key].tolist()
            acc_key = key.replace('_loss', '_acc')
            accs = data[acc_key].tolist() if acc_key in data.files else []
            results[(ep, bs)] = (losses, accs)
    print(f"Loaded {len(results)} cached results from {cache_path}")
    return results


def plot_all(results, output_dir):
    """
    results: dict mapping (epochs, batch) → (test_losses, test_accs)
    Plots 2 figures:
      1) test_loss_all.png  — all loss curves
      2) test_acc_all.png   — all accuracy curves
    """
    os.makedirs(output_dir, exist_ok=True)

    # 按 batch 排序，给每个 batch 分配固定颜色和线型
    unique_batches = sorted(set(b for _, b in results.keys()))
    color_map = {b: _get_color(i) for i, b in enumerate(unique_batches)}
    style_map = {b: _get_style(i) for i, b in enumerate(unique_batches)}

    combos = sorted(results.keys(), key=lambda x: x[1])  # sort by batch

    # ============ Test Loss ============
    fig, ax = plt.subplots(figsize=(16, 10))
    for epochs, batch in combos:
        losses, _ = results[(epochs, batch)]
        x = range(1, len(losses) + 1)
        label = f"ep={epochs} bs={batch}"
        ax.plot(x, losses, color=color_map[batch], linestyle=style_map[batch],
                linewidth=1.2, alpha=0.85, label=label)

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Test Loss (NLL)', fontsize=13)
    ax.set_title(f'EEGNet Test Loss — BCIC IV 2a Subject {SUBJECT}', fontsize=14)
    ax.legend(ncol=3, fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = os.path.join(output_dir, "test_loss_all.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {path}")

    # ============ Test Accuracy ============
    fig, ax = plt.subplots(figsize=(16, 10))
    for epochs, batch in combos:
        _, accs = results[(epochs, batch)]
        x = range(1, len(accs) + 1)
        label = f"ep={epochs} bs={batch}"
        ax.plot(x, accs, color=color_map[batch], linestyle=style_map[batch],
                linewidth=1.2, alpha=0.85, label=label)

    ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.4, linewidth=1,
               label='Chance (25%)')
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Test Accuracy', fontsize=13)
    ax.set_title(f'EEGNet Test Accuracy — BCIC IV 2a Subject {SUBJECT}', fontsize=14)
    ax.legend(ncol=3, fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = os.path.join(output_dir, "test_acc_all.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {path}")


# ============================ Main ============================

def main():
    device = build_device()
    print(f"Device: {device}\n")

    # Load data once
    (X_train, y_train), (X_test, y_test), meta = load_bcic_iv_2a(
        SUBJECT, tmin=TMIN, tmax=TMAX,
    )
    n_ch, n_times, n_cls = meta['n_channels'], meta['n_samples'], meta['n_classes']

    # Load cached results (supports incremental runs)
    results = load_cache(CACHE_FILE)

    combos = list(product(EPOCHS_LIST, BATCH_LIST))
    new_combos = [(e, b) for e, b in combos if (e, b) not in results]
    skipped = len(combos) - len(new_combos)

    print(f"\nGrid: {len(EPOCHS_LIST)} epochs × {len(BATCH_LIST)} batches = {len(combos)} runs")
    print(f"  Cached: {skipped}  |  To run: {len(new_combos)}")
    print(f"Epochs: {EPOCHS_LIST}")
    print(f"Batches: {BATCH_LIST}")
    print(f"Output: {OUTPUT_DIR}/\n")

    for idx, (epochs, batch_size) in enumerate(new_combos):
        run_label = f"ep={epochs} bs={batch_size}"
        print(f"[{idx + 1}/{len(new_combos)}] {run_label}  ———  {epochs} epochs, batch={batch_size}")
        t0 = time.time()

        test_losses, test_accs = run_one(
            epochs, batch_size, X_train, y_train, X_test, y_test,
            n_ch, n_times, n_cls, device,
        )

        elapsed = time.time() - t0
        results[(epochs, batch_size)] = (test_losses, test_accs)
        print(f"  done in {elapsed:.0f}s  |  final test_loss={test_losses[-1]:.4f}  "
              f"test_acc={test_accs[-1]:.2%}\n")

        # Save after each combo — 防止中断丢失
        save_cache(results, CACHE_FILE)

    # Plot all curves
    plot_all(results, OUTPUT_DIR)
    print("Done.")


if __name__ == '__main__':
    main()
