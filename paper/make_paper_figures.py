"""Generate the paper-only figures that aren't already produced by the main
pipeline: system workflow diagram, DWT decomposition illustration (computed
on a real beat), HRV/RR illustration (computed on real record 100 data),
CNN-BiLSTM architecture schematic, and the graphical abstract.

Run from paper/: python make_paper_figures.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pywt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_loader import load_record  # noqa: E402
from preprocessing import clean_signal, segment_beats  # noqa: E402
from utils import BEAT_WINDOW_SEC, FS  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "figures")
NAVY = "#1b3a5c"
TEAL = "#2a9d8f"
MUTED = "#6c7a89"
LIGHT = "#eef3f6"


def box(ax, xy, w, h, text, fc="white", ec=NAVY, fontsize=8.5, lw=1.3):
    b = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                        linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(b)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color="#1a1a1a", wrap=True)
    return b


def arrow(ax, p1, p2, color=MUTED):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12,
                                  linewidth=1.2, color=color))


# ---------------------------------------------------------------------------
# 1. System workflow diagram (Fig. "Overall proposed workflow")
# ---------------------------------------------------------------------------
def make_workflow():
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    box(ax, (0.2, 4.8), 2.1, 1.0, "MIT-BIH ECG\n+ beat annotations\n(PhysioNet)")
    box(ax, (0.2, 3.1), 2.1, 1.0, "Preprocessing:\nbaseline removal +\n0.5-40 Hz bandpass")
    box(ax, (0.2, 1.4), 2.1, 1.0, "Beat segmentation\n+ z-score normalise\n(244 samples/beat)")
    arrow(ax, (1.25, 4.8), (1.25, 4.1))
    arrow(ax, (1.25, 3.1), (1.25, 2.4))

    box(ax, (2.9, 4.6), 2.0, 0.9, "DS1 (train, 22 rec.)\nGroupKFold CV", fc=LIGHT)
    box(ax, (2.9, 0.5), 2.0, 0.9, "DS2 (test, 22 rec.)\nheld out once", fc=LIGHT)
    arrow(ax, (2.3, 2.1), (2.9, 4.1))
    arrow(ax, (2.3, 1.7), (2.9, 1.0))

    y0 = [4.9, 3.0, 1.1]
    labels = ["RR/HRV features\n→ SVM (RBF)", "DWT (db4) features\n→ Random Forest",
              "Raw waveform\n→ 1D-CNN + BiLSTM"]
    for y, lab in zip(y0, labels):
        box(ax, (5.2, y), 2.3, 0.9, lab, fc="white", ec=TEAL)
        arrow(ax, (4.9, 4.95), (5.2, y + 0.45), color=TEAL)

    box(ax, (8.0, 2.65), 1.7, 1.3, "Evaluation on\nDS2:\naccuracy, macro-F1,\nper-class recall", fc=LIGHT, ec=NAVY)
    for y in y0:
        arrow(ax, (7.5, y + 0.45), (8.0, 3.3), color=NAVY)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "workflow.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. DWT decomposition illustration (real beat, real pywt.wavedec)
# ---------------------------------------------------------------------------
def make_dwt_figure():
    sig, fields, ann = load_record(100)
    cleaned = clean_signal(sig[:, fields["sig_name"].index("MLII")], fs=fields["fs"])
    beats, labels, _, _ = segment_beats(cleaned, ann.sample, ann.symbol, fs=fields["fs"])
    beat = beats[np.where(labels == "N")[0][0]]

    coeffs = pywt.wavedec(beat, "db4", level=4)
    names = ["cA4"] + [f"cD{l}" for l in range(4, 0, -1)]

    fig, axes = plt.subplots(len(coeffs) + 1, 1, figsize=(5.0, 7.2), sharex=False)
    before = int(BEAT_WINDOW_SEC[0] * FS)
    t = (np.arange(len(beat)) - before) / FS
    axes[0].plot(t, beat, color=NAVY)
    axes[0].set_title("Input beat (normalised)", fontsize=9)
    for ax, c, name in zip(axes[1:], coeffs, names):
        ax.plot(c, color=TEAL)
        ax.set_title(f"{name} ({len(c)} coeffs)", fontsize=9)
    for ax in axes:
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dwt_decomposition.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. HRV / RR-interval illustration (real record 100 segment)
# ---------------------------------------------------------------------------
def make_hrv_figure():
    from utils import SYMBOL_TO_AAMI

    sig, fields, ann = load_record(100)
    cleaned = clean_signal(sig[:, fields["sig_name"].index("MLII")], fs=fields["fs"])
    beat_samples = [s for s, sym in zip(ann.sample, ann.symbol) if sym in SYMBOL_TO_AAMI]

    n_beats_shown = 5
    r_samples = beat_samples[:n_beats_shown]
    seg_end = r_samples[-1] + int(0.4 * fields["fs"])
    t = np.arange(seg_end) / fields["fs"]

    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    ax.plot(t, cleaned[:seg_end], color=NAVY, linewidth=1)
    ax.scatter(np.array(r_samples) / fields["fs"], cleaned[r_samples], color=TEAL, zorder=5, s=30,
               label="annotated R-peak")
    y_top = cleaned[:seg_end].max() * 1.35
    for i in range(len(r_samples) - 1):
        x1, x2 = r_samples[i] / fields["fs"], r_samples[i + 1] / fields["fs"]
        y = y_top * (1.0 if i % 2 == 0 else 1.18)
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1))
        ax.text((x1 + x2) / 2, y + y_top * 0.04, f"RR={x2 - x1:.2f}s", ha="center", fontsize=7, color=MUTED)
    ax.set_ylim(top=y_top * 1.45)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mV")
    ax.set_title("RR-interval extraction from consecutive R-peaks (record 100)", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "hrv_illustration.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. CNN-BiLSTM architecture schematic (matches src/model_cnn_bilstm.py)
# ---------------------------------------------------------------------------
def make_cnn_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 2.6)
    ax.axis("off")

    stages = [
        "Input\n(1x244)",
        "Conv1D 16@k7\n+ ReLU\n+ MaxPool(2)",
        "Conv1D 32@k5\n+ ReLU\n+ MaxPool(2)",
        "BiLSTM\nhidden=64\n(bidirectional)",
        "Dropout\np=0.3",
        "Dense\n(128→5)",
        "Softmax\n(N,S,V,F,Q)",
    ]
    n = len(stages)
    w, gap = 1.2, 0.28
    x = 0.15
    for i, s in enumerate(stages):
        fc = TEAL if "Conv1D" in s else (NAVY if "BiLSTM" in s else "white")
        tc = "white" if fc != "white" else "#1a1a1a"
        b = box(ax, (x, 0.7), w, 1.2, s, fc=fc, ec=NAVY, fontsize=7.6)
        if tc == "white":
            ax.texts[-1].set_color("white")
        if i < n - 1:
            arrow(ax, (x + w, 1.3), (x + w + gap, 1.3))
        x += w + gap
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cnn_bilstm_architecture.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Graphical abstract (distinct style, 4-stage horizontal flow, real metrics)
# ---------------------------------------------------------------------------
def make_graphical_abstract():
    fig, ax = plt.subplots(figsize=(13, 4.0))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.0)
    ax.axis("off")
    ax.set_facecolor("white")

    ax.text(6.5, 3.75, "ECG Arrhythmia Classification Using Advanced Temporal Pattern Recognition",
            ha="center", fontsize=13, weight="bold", color=NAVY)
    ax.text(6.55, 3.35, "STAGE 3: Three Parallel Classification Approaches", ha="center", fontsize=8.5,
            weight="bold", color=NAVY)

    y_mid = 1.55

    # Stage 1
    box(ax, (0.2, y_mid - 0.7), 2.1, 1.4, "STAGE 1\nData Acquisition\n\nMIT-BIH Arrhythmia DB\n44 records, 360 Hz\n+ beat annotations",
        fc=LIGHT, ec=NAVY, fontsize=8)
    arrow(ax, (2.3, y_mid), (2.75, y_mid))

    # Stage 2
    box(ax, (2.8, y_mid - 0.7), 2.1, 1.4, "STAGE 2\nPreprocessing\n\nBaseline removal\n0.5-40 Hz bandpass\nBeat segmentation\nZ-score normalise",
        fc=LIGHT, ec=NAVY, fontsize=8)
    arrow(ax, (4.9, y_mid), (5.35, y_mid))

    # Stage 3 - 3 branches
    box(ax, (5.4, y_mid + 0.45), 2.3, 0.75, "Branch A: RR/HRV → SVM", fc="white", ec=TEAL, fontsize=7.8)
    box(ax, (5.4, y_mid - 0.375), 2.3, 0.75, "Branch B: DWT(db4) → Random Forest", fc="white", ec=TEAL, fontsize=7.8)
    box(ax, (5.4, y_mid - 1.2), 2.3, 0.75, "Branch C: 1D-CNN → BiLSTM → Softmax", fc="white", ec=TEAL, fontsize=7.8)
    for y in (y_mid + 0.825, y_mid, y_mid - 0.825):
        arrow(ax, (7.7, y), (8.15, y_mid), color=TEAL)

    # Stage 4
    box(ax, (8.2, y_mid - 0.7), 2.2, 1.4,
        "STAGE 4\nEvaluation\n\n5-class AAMI scheme\nN, S, V, F, Q\nDS2 held-out test set",
        fc=LIGHT, ec=NAVY, fontsize=8)
    arrow(ax, (10.4, y_mid), (10.85, y_mid))

    # Results panel (real numbers)
    box(ax, (10.9, y_mid - 0.8), 2.0, 1.6,
        "Macro-F1 (DS2)\nSVM: 0.362\nRF: 0.334\nCNN-BiLSTM: 0.183\n\nAccuracy\nRF: 0.884 (best)",
        fc="white", ec=NAVY, fontsize=7.6)

    ax.text(6.5, 0.15,
            "Patient-independent DS1/DS2 split (de Chazal et al.) — no patient's beats appear in both train and test",
            ha="center", fontsize=8, color=MUTED, style="italic")

    fig.savefig(os.path.join(OUT, "graphical_abstract.png"), dpi=220, facecolor="white", bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "graphical_abstract.svg"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    make_workflow()
    print("workflow.png done")
    make_dwt_figure()
    print("dwt_decomposition.png done")
    make_hrv_figure()
    print("hrv_illustration.png done")
    make_cnn_architecture()
    print("cnn_bilstm_architecture.png done")
    make_graphical_abstract()
    print("graphical_abstract.png/.svg done")
