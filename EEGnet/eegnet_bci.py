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


@torch.no_grad()
def detailed_eval(model, loader, device):
    """返回 all_preds, all_targets — 用于计算混淆矩阵和 p 值."""
    model.eval()
    all_preds, all_targets = [], []
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        all_preds.append(output.argmax(1).cpu().numpy())
        all_targets.append(target.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def compute_metrics(y_true, y_pred, class_names):
    """
    计算四分类指标:
      - 混淆矩阵
      - 每类准确率 (recall)
      - 整体准确率 + 二项检验 p 值 (H0: acc = 25%)
    """
    from scipy.stats import binomtest

    n_total = len(y_true)
    correct = (y_pred == y_true).sum()
    acc = correct / n_total

    # p 值: P(X ≥ correct | X ~ Binomial(n_total, 0.25))
    result = binomtest(correct, n=n_total, p=0.25, alternative='greater')
    p_value = result.pvalue

    # 混淆矩阵 & 每类召回率
    n_cls = len(class_names)
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class_acc = {}
    for i, name in enumerate(class_names):
        row_sum = cm[i].sum()
        per_class_acc[name] = cm[i, i] / row_sum if row_sum > 0 else 0.0

    return {
        'accuracy': acc,
        'correct': correct,
        'total': n_total,
        'p_value': p_value,
        'confusion_matrix': cm,
        'per_class_acc': per_class_acc,
    }


def print_metrics(metrics, class_names, title=""):
    """格式化打印分类报告."""
    m = metrics
    stars = " ***" if m['p_value'] < 0.001 else (
        " **" if m['p_value'] < 0.01 else (" *" if m['p_value'] < 0.05 else ""))
    print(f"\n{'='*56}")
    if title:
        print(f"  {title}")
    print(f"  Overall Accuracy: {m['accuracy']:.2%} ({m['correct']}/{m['total']})")
    print(f"  p-value (vs chance 25%): {m['p_value']:.2e}{stars}")
    print(f"  {'Class':>12s}  {'Recall':>8s}  #samples")
    print(f"  {'-'*36}")
    for i, name in enumerate(class_names):
        row = m['confusion_matrix'][i]
        print(f"  {name:>12s}  {m['per_class_acc'][name]:7.2%}  "
              f"({row.sum()})")
    print(f"{'='*56}\n")
    return m


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

    # Final detailed evaluation
    y_pred, y_true = detailed_eval(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, CLASS_NAMES)

    return test_losses, test_accs, metrics, model


# ======================= Plotting =======================

# 颜色和线型：动态分配，支持任意 batch 值
BATCH_COLORS = plt.cm.tab10.colors  # 最多 10 种颜色


def _get_color(batch_idx):
    return BATCH_COLORS[batch_idx % len(BATCH_COLORS)]

def _get_style(batch_idx):
    return ['-', '--', '-.', ':', (0, (3, 1, 1, 1))][batch_idx % 5]


# ======================= Caching =======================

def _cache_fingerprint():
    """当前参数的唯一标识 — 参数变了缓存自动失效."""
    import json
    return json.dumps([EPOCHS_LIST, BATCH_LIST], sort_keys=True)


def save_cache(results, cache_path):
    """保存结果到 npz 文件, 附带参数指纹."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    flat = {'__fingerprint': _cache_fingerprint()}
    for (ep, bs), (losses, accs, metrics) in results.items():
        flat[f"{ep}_{bs}_loss"] = np.array(losses, dtype=np.float32)
        flat[f"{ep}_{bs}_acc"] = np.array(accs, dtype=np.float32)
        flat[f"{ep}_{bs}_pval"] = metrics['p_value']
        flat[f"{ep}_{bs}_cm"] = metrics['confusion_matrix']
        for i, name in enumerate(CLASS_NAMES):
            flat[f"{ep}_{bs}_recall_{name}"] = metrics['per_class_acc'][name]
    np.savez(cache_path, **flat)
    print(f"Cache saved → {cache_path}")


def load_cache(cache_path):
    """加载缓存; 参数改动时自动清除旧缓存."""
    if not os.path.exists(cache_path):
        return {}
    data = np.load(cache_path, allow_pickle=True)

    # 指纹不匹配 → 清缓存
    if '__fingerprint' not in data.files or \
       str(data['__fingerprint']) != _cache_fingerprint():
        print("Cache params changed — clearing old cache")
        os.remove(cache_path)
        return {}

    results = {}
    for key in data.files:
        if key.startswith('__'):
            continue
        if key.endswith('_loss'):
            parts = key.replace('_loss', '').split('_', 1)
            ep, bs = int(parts[0]), int(parts[1])
            losses = data[key].tolist()
            accs = data[f"{ep}_{bs}_acc"].tolist()
            metrics = {
                'accuracy': accs[-1],
                'p_value': float(data.get(f"{ep}_{bs}_pval", np.nan)),
                'confusion_matrix': data.get(f"{ep}_{bs}_cm", np.zeros((4, 4))),
                'per_class_acc': {},
                'correct': 0,
                'total': 0,
            }
            for i, name in enumerate(CLASS_NAMES):
                k = f"{ep}_{bs}_recall_{name}"
                if k in data.files:
                    metrics['per_class_acc'][name] = float(data[k])
            results[(ep, bs)] = (losses, accs, metrics)
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
        losses = results[(epochs, batch)][0]
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
        accs = results[(epochs, batch)][1]
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


# ======================= Topomap =======================


def plot_spatial_filters(model, output_dir, tag="", F1=8, D=2):
    """
    绘制 EEGNet Block1 空间滤波器 (depthwise conv) 的脑地形图。

    每个子图对应一个空间滤波器，显示 22 个电极上的权重分布。
    共 D×F1 = 16 张图（D 行，F1 列）。
    """
    import mne

    # BCIC IV 2a 电极名称 → 10-20 标准位
    ch_names = [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
        "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "CP3", "CP1", "CPz", "CP2", "CP4",
        "P1", "Pz", "P2", "POz",
    ]

    # 创建 MNE Info（需 montage）
    montage = mne.channels.make_standard_montage("standard_1005")
    info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types="eeg")
    info.set_montage(montage)

    # 提取空间滤波器权重: block1[1] → shape (D*F1, 1, n_channels, 1)
    spatial_weights = model.block1[1].weight.detach().cpu().numpy()
    n_filters = spatial_weights.shape[0]  # D * F1

    # 每组 (F1=8 列, D=2 行)
    fig, axes = plt.subplots(D, F1, figsize=(F1 * 2.5, D * 2.5))
    # 确保 axes 可迭代
    if D == 1:
        axes = axes[np.newaxis, :]
    if F1 == 1:
        axes = axes[:, np.newaxis]

    vmax = np.abs(spatial_weights).max()

    for d in range(D):
        for f in range(F1):
            idx = d * F1 + f
            weights_1d = spatial_weights[idx, 0, :, 0]  # (22,)
            ax = axes[d, f]
            mne.viz.plot_topomap(
                weights_1d, info, axes=ax, show=False,
                vlim=(-vmax, vmax), contours=0,
                cmap='RdBu_r', sensors=True,
            )
            ax.set_title(f"F={f + 1}  D={d + 1}", fontsize=9)

    fig.suptitle(f"EEGNet Spatial Filters (Depthwise Conv){tag}",
                 fontsize=13, y=1.02)
    fig.tight_layout()

    filename = f"topomap{tag.replace(' ', '_')}.png" if tag else "topomap.png"
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {path}")


# ======================= Waveform plots =======================


def plot_raw_vs_filtered(output_dir, subject=1, n_examples=3):
    """
    绘制原始信号 vs 带通滤波后信号的波形图。

    每类选 n_examples 个 trial，对比展示 22 通道的原始波形（上排）和
    2-40 Hz 滤波后波形（下排），保存到 output_dir/。
    """
    import glob

    data_dir = os.path.join(
        os.path.dirname(__file__), "mne_data", "MNE-bnci-data",
        "~bci", "database", "001-2014",
    )
    mat_path = glob.glob(os.path.join(data_dir, f"A{subject:02d}T.mat"))
    if not mat_path:
        print("  [waveform] .mat file not found, skipping")
        return
    mat_path = mat_path[0]

    data = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    run_array = data["data"] if isinstance(data["data"], np.ndarray) else [data["data"]]

    sfreq = SFREQ
    ch_names = EEG_CHANNELS

    # 显示窗口: cue 前 0.5s → cue 后 3.5s (总 4s = 1000 样本)
    cue_sample = int(2.0 * sfreq)
    disp_start = cue_sample - int(0.5 * sfreq)   # 375
    disp_end = cue_sample + int(3.5 * sfreq)      # 1375
    n_disp = disp_end - disp_start

    # 收集每类 trials
    trials_by_class = {i: [] for i in range(4)}
    for run in run_array:
        if len(run.trial) == 0:
            continue
        trials = run.trial.ravel().astype(int)
        labels = run.y.ravel().astype(int)
        eeg_raw = run.X.astype(np.float64)
        eeg_filt = _bandpass_filter(eeg_raw, LOWCUT, HIGHCUT, sfreq)
        for trial_pos, label in zip(trials, labels):
            t0 = trial_pos - 1
            raw_seg = eeg_raw[t0 + disp_start:t0 + disp_end, :22]
            filt_seg = eeg_filt[t0 + disp_start:t0 + disp_end, :22]
            if raw_seg.shape[0] >= n_disp and filt_seg.shape[0] >= n_disp:
                trials_by_class[label - 1].append((raw_seg, filt_seg))

    os.makedirs(output_dir, exist_ok=True)
    t = np.arange(n_disp) / sfreq  # 秒

    for cls_idx, name in enumerate(CLASS_NAMES):
        samples = trials_by_class[cls_idx][:n_examples]
        if not samples:
            continue

        fig, axes = plt.subplots(
            2, n_examples, figsize=(4 * n_examples, 10),
            sharex=True, sharey='row',
        )
        if n_examples == 1:
            axes = axes[:, np.newaxis]

        for col, (raw_seg, filt_seg) in enumerate(samples):
            # 上排: 原始信号
            ax_raw = axes[0, col]
            for ch in range(22):
                ax_raw.plot(t, raw_seg[:, ch] + ch * 30, linewidth=0.4)
            ax_raw.set_title(f"Raw — Trial {col + 1}" if col > 0 else f"Raw")
            ax_raw.set_ylim(-80, 22 * 30 + 80)
            if col == 0:
                ax_raw.set_ylabel("Channel (offset)")

            # 下排: 滤波后
            ax_filt = axes[1, col]
            for ch in range(22):
                ax_filt.plot(t, filt_seg[:, ch] + ch * 30, linewidth=0.4)
            ax_filt.set_title(f"Filtered {LOWCUT}–{HIGHCUT} Hz — Trial {col + 1}" if col > 0 else f"Filtered {LOWCUT}–{HIGHCUT} Hz")
            ax_filt.set_xlabel("Time (s)")
            if col == 0:
                ax_filt.set_ylabel("Channel (offset)")

        # 添加 cue 线
        for row in range(2):
            for col in range(n_examples):
                axes[row, col].axvline(x=2.0, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
                axes[row, col].axvline(x=3.25, color='gray', linestyle=':', linewidth=0.6, alpha=0.4)

        fig.suptitle(f"Subject {subject} — {name}  (red=cue onset, gray=MI start)",
                     fontsize=12, y=1.01)
        fig.tight_layout()
        path = os.path.join(output_dir, f"waveform_{name}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved → {path}")


# ============================ Main ============================

def main():
    device = build_device()
    print(f"Device: {device}\n")

    # ---- 创建带时间戳的输出目录 ----
    run_dir = os.path.join(OUTPUT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"Output dir: {run_dir}/\n")

    # ---- 画原始 vs 滤波波形图 ----
    print("Plotting raw vs filtered waveforms …")
    plot_raw_vs_filtered(run_dir, subject=SUBJECT)

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
    print(f"Output: {run_dir}/\n")

    for idx, (epochs, batch_size) in enumerate(new_combos):
        run_label = f"ep={epochs} bs={batch_size}"
        print(f"[{idx + 1}/{len(new_combos)}] {run_label}  ———  {epochs} epochs, batch={batch_size}")
        t0 = time.time()

        test_losses, test_accs, metrics, _model = run_one(
            epochs, batch_size, X_train, y_train, X_test, y_test,
            n_ch, n_times, n_cls, device,
        )

        elapsed = time.time() - t0
        results[(epochs, batch_size)] = (test_losses, test_accs, metrics)
        print(f"  done in {elapsed:.0f}s  |  final test_loss={test_losses[-1]:.4f}  "
              f"test_acc={test_accs[-1]:.2%}")
        print_metrics(metrics, CLASS_NAMES, title=run_label)
        plot_spatial_filters(_model, run_dir, tag=f"_{run_label}")

        # Save after each combo — 防止中断丢失
        save_cache(results, CACHE_FILE)

    # ---- 所有组合汇总表 ----
    print(f"\n{'='*80}")
    print(f"{'Summary':^80}")
    print(f"{'='*80}")
    print(f"{'Combo':>20s}  {'Test Acc':>9s}  {'p-value':>10s}  {'Signif':>6s}  "
          f"{'left_hand':>9s}  {'right_hand':>9s}  {'feet':>9s}  {'tongue':>9s}")
    print(f"{'-'*80}")
    for (ep, bs), (_, _, m) in sorted(results.items(), key=lambda x: x[0][1]):
        sig = "***" if m['p_value'] < 0.001 else ("**" if m['p_value'] < 0.01 else ("*" if m['p_value'] < 0.05 else ""))
        pca = m['per_class_acc']
        print(f"{f'ep={ep} bs={bs}':>20s}  {m['accuracy']:8.2%}  {m['p_value']:10.2e}  {sig:>6s}  "
              f"{pca['left_hand']:8.2%}  {pca['right_hand']:8.2%}  {pca['feet']:8.2%}  {pca['tongue']:8.2%}")
    print(f"{'='*80}")

    # Plot all curves
    plot_all(results, run_dir)
    print("Done.")


if __name__ == '__main__':
    main()
