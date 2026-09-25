#!/usr/bin/env python3
"""
Analyze and visualize the probability-rank paradox between two fine-tuned models.

Works in two modes:
  1. Full analysis (requires per-token data from collect_per_token_details.py):
       python visualize_prob_rank_paradox.py \
         --m1_data analysis/per_token_M1.jsonl \
         --m2_data analysis/per_token_M2.jsonl \
         --output_dir analysis/paradox_figures

  2. Summary-only + conceptual illustration (no per-token data needed):
       python visualize_prob_rank_paradox.py \
         --m1_summary analysis/.../M1_.../summary.txt \
         --m2_summary analysis/.../M2_.../summary.txt \
         --output_dir analysis/paradox_figures
"""

import json
import re
import argparse
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_per_token_data(path: str) -> List[Dict]:
    """Load per-token data from a single JSONL file or a directory of per-task JSONL files.

    Supports both formats:
      - Single JSONL (from collect_per_token_details.py): each line has a "task" field.
      - Directory of JSONLs (from analyze_token_prob_rank.py _raw/): filename is the task.
    """
    p = Path(path)
    if p.is_dir():
        data = []
        for jsonl_file in sorted(p.glob("*.jsonl")):
            task_name = jsonl_file.stem
            with open(jsonl_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        if "task" not in rec:
                            rec["task"] = task_name
                        data.append(rec)
        return data
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def parse_summary_file(path: str) -> Dict[str, Dict[str, float]]:
    """Parse summary.txt -> {task: {avg_prob, avg_rank, n_samples}}."""
    results = {}
    current_task = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            m = re.match(r"^(\S+):$", line)
            if m and not line.startswith("Model") and not line.startswith("Token"):
                current_task = m.group(1)
                results[current_task] = {}
                continue
            if current_task:
                m_prob = re.search(r"Avg Probability:\s+([\d.]+)", line)
                m_rank = re.search(r"Avg Rank:\s+([\d.]+)", line)
                m_samples = re.search(r"Samples:\s+(\d+)", line)
                if m_prob:
                    results[current_task]["avg_prob"] = float(m_prob.group(1))
                if m_rank:
                    results[current_task]["avg_rank"] = float(m_rank.group(1))
                if m_samples:
                    results[current_task]["n_samples"] = int(m_samples.group(1))
    return results


def group_by_task(data: List[Dict]) -> Dict[str, List[Dict]]:
    groups = defaultdict(list)
    for rec in data:
        groups[rec["task"]].append(rec)
    return dict(groups)


def match_samples(data_m1: List[Dict], data_m2: List[Dict]) -> List[Tuple[Dict, Dict]]:
    """Match samples by sample_id between two model datasets."""
    m2_index = {rec["sample_id"]: rec for rec in data_m2}
    pairs = []
    for rec1 in data_m1:
        sid = rec1["sample_id"]
        if sid in m2_index:
            pairs.append((rec1, m2_index[sid]))
    return pairs


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_rank_stats(data: List[Dict]) -> Dict[str, Any]:
    """Compute aggregate rank statistics from per-token data."""
    all_ranks = []
    all_probs = []
    for rec in data:
        all_ranks.extend(rec["token_ranks"])
        all_probs.extend(rec["token_probs"])
    all_ranks = np.array(all_ranks)
    all_probs = np.array(all_probs)

    thresholds = [1, 5, 10, 50, 100, 500, 1000]
    rank_cdf = {k: float(np.mean(all_ranks <= k)) for k in thresholds}

    return {
        "total_tokens": len(all_ranks),
        "mean_rank": float(np.mean(all_ranks)),
        "median_rank": float(np.median(all_ranks)),
        "p90_rank": float(np.percentile(all_ranks, 90)),
        "p99_rank": float(np.percentile(all_ranks, 99)),
        "mean_prob": float(np.mean(all_probs)),
        "rank_cdf": rank_cdf,
        "all_ranks": all_ranks,
        "all_probs": all_probs,
    }


def find_paradox_examples(
    pairs: List[Tuple[Dict, Dict]], n: int = 20
) -> List[Dict]:
    """
    Find samples where M2 has higher avg_prob but much worse avg_rank than M1.
    Score = (rank_m2 / rank_m1) * (prob_m2 / prob_m1) to capture both aspects.
    """
    scored = []
    for rec1, rec2 in pairs:
        if rec1["avg_rank"] < 1e-6:
            continue
        rank_ratio = rec2["avg_rank"] / rec1["avg_rank"]
        prob_diff = rec2["avg_prob"] - rec1["avg_prob"]
        if prob_diff > 0 and rank_ratio > 2:
            scored.append({
                "sample_id": rec1["sample_id"],
                "task": rec1["task"],
                "output": rec1["output"],
                "m1_avg_prob": rec1["avg_prob"],
                "m1_avg_rank": rec1["avg_rank"],
                "m2_avg_prob": rec2["avg_prob"],
                "m2_avg_rank": rec2["avg_rank"],
                "rank_ratio": rank_ratio,
                "prob_diff": prob_diff,
                "rec1": rec1,
                "rec2": rec2,
            })
    scored.sort(key=lambda x: x["rank_ratio"], reverse=True)
    return scored[:n]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {"m1": "#2196F3", "m2": "#FF5722", "accent": "#4CAF50", "gray": "#9E9E9E"}


def plot_task_comparison(
    summary_m1: Dict, summary_m2: Dict, output_path: str,
    m1_name: str = "M1 (LST)", m2_name: str = "M2 (FFT)",
):
    """Side-by-side bar charts: probability (left) and rank (right) per task."""
    tasks = sorted(set(summary_m1.keys()) & set(summary_m2.keys()))
    if not tasks:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    x = np.arange(len(tasks))
    w = 0.35

    probs_m1 = [summary_m1[t]["avg_prob"] for t in tasks]
    probs_m2 = [summary_m2[t]["avg_prob"] for t in tasks]
    ranks_m1 = [summary_m1[t]["avg_rank"] for t in tasks]
    ranks_m2 = [summary_m2[t]["avg_rank"] for t in tasks]

    ax1.bar(x - w / 2, probs_m1, w, label=m1_name, color=COLORS["m1"], alpha=0.85)
    ax1.bar(x + w / 2, probs_m2, w, label=m2_name, color=COLORS["m2"], alpha=0.85)
    ax1.set_ylabel("Average Probability", fontsize=13)
    ax1.set_title("Average Token Probability by Task", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, rotation=45, ha="right", fontsize=10)
    ax1.legend(fontsize=11)
    ax1.set_ylim(0, 1.05)
    for i, (v1, v2) in enumerate(zip(probs_m1, probs_m2)):
        winner = "m2" if v2 > v1 else "m1"
        if winner == "m2":
            ax1.annotate(
                f"+{v2 - v1:.3f}", (i + w / 2, v2), ha="center", va="bottom",
                fontsize=8, color=COLORS["m2"], fontweight="bold",
            )

    ax2.bar(x - w / 2, ranks_m1, w, label=m1_name, color=COLORS["m1"], alpha=0.85)
    ax2.bar(x + w / 2, ranks_m2, w, label=m2_name, color=COLORS["m2"], alpha=0.85)
    ax2.set_ylabel("Average Rank (log scale)", fontsize=13)
    ax2.set_title("Average Token Rank by Task", fontsize=14, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tasks, rotation=45, ha="right", fontsize=10)
    ax2.legend(fontsize=11)
    ax2.set_yscale("log")
    for i, (v1, v2) in enumerate(zip(ranks_m1, ranks_m2)):
        if v2 > v1 * 2:
            ax2.annotate(
                f"{v2 / v1:.0f}x", (i + w / 2, v2), ha="center", va="bottom",
                fontsize=9, color=COLORS["m2"], fontweight="bold",
            )

    fig.suptitle(
        "Probability-Rank Paradox: Higher Prob Does NOT Imply Better Rank",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_rank_distribution(
    stats_m1: Dict, stats_m2: Dict, output_path: str,
    m1_name: str = "M1 (LST)", m2_name: str = "M2 (FFT)",
    task_label: str = "",
):
    """Plot rank CDF and histogram comparing two models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ranks_m1 = stats_m1["all_ranks"]
    ranks_m2 = stats_m2["all_ranks"]

    # CDF
    max_rank = int(max(np.percentile(ranks_m1, 99.5), np.percentile(ranks_m2, 99.5)))
    max_rank = max(max_rank, 100)
    eval_points = np.logspace(0, np.log10(max_rank + 1), 500)
    cdf_m1 = np.array([np.mean(ranks_m1 <= r) for r in eval_points])
    cdf_m2 = np.array([np.mean(ranks_m2 <= r) for r in eval_points])

    ax1.plot(eval_points, cdf_m1, color=COLORS["m1"], linewidth=2.5, label=m1_name)
    ax1.plot(eval_points, cdf_m2, color=COLORS["m2"], linewidth=2.5, label=m2_name)
    ax1.fill_between(
        eval_points, cdf_m1, cdf_m2,
        where=cdf_m1 > cdf_m2, alpha=0.15, color=COLORS["m1"],
    )
    ax1.set_xscale("log")
    ax1.set_xlabel("Rank Threshold (k)", fontsize=13)
    ax1.set_ylabel("Fraction of Tokens with Rank ≤ k", fontsize=13)
    title_suffix = f" — {task_label}" if task_label else ""
    ax1.set_title(f"Rank CDF{title_suffix}", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=12, loc="lower right")
    ax1.axhline(y=0.95, color=COLORS["gray"], linestyle="--", alpha=0.5)
    ax1.annotate("95%", (1.2, 0.955), fontsize=10, color=COLORS["gray"])
    ax1.set_ylim(0, 1.02)
    ax1.grid(True, alpha=0.3)

    # Histogram with log bins
    bin_edges = [0.5, 1.5, 5.5, 10.5, 50.5, 100.5, 500.5, 1000.5, 10000.5]
    bin_labels = ["1", "2-5", "6-10", "11-50", "51-100", "101-500", "501-1K", "1K+"]
    counts_m1, _ = np.histogram(ranks_m1, bins=bin_edges)
    counts_m2, _ = np.histogram(ranks_m2, bins=bin_edges)
    frac_m1 = counts_m1 / len(ranks_m1)
    frac_m2 = counts_m2 / len(ranks_m2)

    x = np.arange(len(bin_labels))
    w = 0.35
    ax2.bar(x - w / 2, frac_m1, w, label=m1_name, color=COLORS["m1"], alpha=0.85)
    ax2.bar(x + w / 2, frac_m2, w, label=m2_name, color=COLORS["m2"], alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bin_labels, fontsize=10)
    ax2.set_xlabel("Rank Bin", fontsize=13)
    ax2.set_ylabel("Fraction of Tokens", fontsize=13)
    ax2.set_title(f"Rank Distribution{title_suffix}", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, axis="y")

    for i in range(len(bin_labels)):
        if frac_m2[i] > frac_m1[i] and frac_m2[i] > 0.01:
            ratio = frac_m2[i] / max(frac_m1[i], 1e-6)
            if ratio > 1.5:
                ax2.annotate(
                    f"{ratio:.1f}x", (i + w / 2, frac_m2[i]),
                    ha="center", va="bottom", fontsize=8,
                    color=COLORS["m2"], fontweight="bold",
                )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_sample_scatter(
    pairs: List[Tuple[Dict, Dict]], output_path: str,
    m1_name: str = "M1 (LST)", m2_name: str = "M2 (FFT)",
):
    """Scatter: each dot is a sample, showing (avg_prob, avg_rank) for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    tasks = sorted(set(p[0]["task"] for p in pairs))
    task_colors = plt.cm.Set2(np.linspace(0, 1, len(tasks)))
    task_color_map = dict(zip(tasks, task_colors))

    for ax, model_idx, model_name in [(axes[0], 0, m1_name), (axes[1], 1, m2_name)]:
        for task in tasks:
            task_pairs = [(p[model_idx]["avg_prob"], p[model_idx]["avg_rank"])
                          for p in pairs if p[0]["task"] == task]
            if not task_pairs:
                continue
            probs, ranks = zip(*task_pairs)
            ax.scatter(
                probs, ranks, c=[task_color_map[task]], label=task,
                alpha=0.5, s=15, edgecolors="none",
            )
        ax.set_xlabel("Average Token Probability", fontsize=12)
        ax.set_ylabel("Average Token Rank (log)", fontsize=12)
        ax.set_yscale("log")
        ax.set_title(model_name, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1.05)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=min(5, len(tasks)),
        fontsize=9, bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Per-Sample Average Prob vs Rank", fontsize=15, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_case_study(
    example: Dict, output_path: str,
    m1_name: str = "M1 (LST)", m2_name: str = "M2 (FFT)",
):
    """
    Token-level comparison for a specific paradox example.
    Shows each token colored by its rank under both models.
    """
    rec1, rec2 = example["rec1"], example["rec2"]
    tokens = rec1["token_strs"]
    ranks_m1 = np.array(rec1["token_ranks"])
    ranks_m2 = np.array(rec2["token_ranks"])
    probs_m1 = np.array(rec1["token_probs"])
    probs_m2 = np.array(rec2["token_probs"])

    n = len(tokens)
    max_show = min(n, 80)
    tokens = tokens[:max_show]
    ranks_m1 = ranks_m1[:max_show]
    ranks_m2 = ranks_m2[:max_show]
    probs_m1 = probs_m1[:max_show]
    probs_m2 = probs_m2[:max_show]

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.6], hspace=0.35)

    rank_cmap = LinearSegmentedColormap.from_list(
        "rank_cmap", ["#4CAF50", "#FFEB3B", "#FF5722", "#B71C1C"], N=256
    )
    rank_norm_max = max(float(np.max(ranks_m1)), float(np.max(ranks_m2)), 100)
    log_norm_max = np.log10(rank_norm_max + 1)

    for panel_idx, (ranks, probs, name) in enumerate([
        (ranks_m1, probs_m1, m1_name), (ranks_m2, probs_m2, m2_name)
    ]):
        ax = fig.add_subplot(gs[panel_idx])
        ax.set_xlim(-0.5, max_show - 0.5)
        ax.set_ylim(0, 1)

        for i in range(max_show):
            log_rank = np.log10(ranks[i] + 1) / log_norm_max
            color = rank_cmap(min(log_rank, 1.0))
            bar_height = probs[i]
            ax.bar(i, bar_height, width=0.9, color=color, edgecolor="none", alpha=0.85)
            if ranks[i] > 10:
                ax.text(
                    i, bar_height + 0.02, f"r{int(ranks[i])}",
                    ha="center", va="bottom", fontsize=6, color="#B71C1C",
                    fontweight="bold", rotation=90,
                )

        ax.set_ylabel("Token Prob", fontsize=11)
        ax.set_title(
            f"{name}  —  avg_prob={np.mean(probs):.4f}  avg_rank={np.mean(ranks):.1f}",
            fontsize=12, fontweight="bold",
        )
        ax.set_xticks(range(max_show))
        ax.set_xticklabels(
            [t.replace("\n", "\\n") for t in tokens],
            fontsize=6, rotation=90,
        )
        ax.grid(True, alpha=0.2, axis="y")

    # Rank difference panel
    ax3 = fig.add_subplot(gs[2])
    rank_diff = ranks_m2 - ranks_m1
    colors = [COLORS["m2"] if d > 0 else COLORS["m1"] for d in rank_diff]
    ax3.bar(range(max_show), rank_diff, width=0.9, color=colors, alpha=0.7)
    ax3.axhline(y=0, color="black", linewidth=0.8)
    ax3.set_ylabel("Rank Diff\n(M2 − M1)", fontsize=10)
    ax3.set_xlabel("Token Position", fontsize=11)
    ax3.set_title("Rank Difference (positive = M2 worse)", fontsize=12, fontweight="bold")
    ax3.set_xticks(range(max_show))
    ax3.set_xticklabels(
        [t.replace("\n", "\\n") for t in tokens],
        fontsize=6, rotation=90,
    )
    ax3.grid(True, alpha=0.2, axis="y")

    fig.suptitle(
        f"Case Study: {example['task']}  |  sample_id: {example['sample_id']}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_conceptual_illustration(
    output_path: str,
    m1_name: str = "M1 (LST)", m2_name: str = "M2 (FFT)",
):
    """
    Synthetic illustration of the prob-rank paradox using toy distributions.
    No real data needed — purely pedagogical.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)

    # ---- Panel (a): Easy token — both models do well ----
    ax_a = fig.add_subplot(gs[0, 0])
    labels = ["Token A\n(correct)", "Token B", "Token C", "Token D", "Others"]
    m1_probs = [0.70, 0.10, 0.07, 0.05, 0.08]
    m2_probs = [0.93, 0.03, 0.01, 0.01, 0.02]
    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax_a.bar(x - w / 2, m1_probs, w, label=m1_name, color=COLORS["m1"], alpha=0.85)
    bars2 = ax_a.bar(x + w / 2, m2_probs, w, label=m2_name, color=COLORS["m2"], alpha=0.85)
    ax_a.set_ylabel("Probability", fontsize=12)
    ax_a.set_title(
        '(a) "Easy" Token — Both Models Confident',
        fontsize=13, fontweight="bold",
    )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, fontsize=10)
    ax_a.legend(fontsize=10)
    ax_a.set_ylim(0, 1.1)
    ax_a.annotate(
        f"{m1_name}: prob=0.70, rank=1\n{m2_name}: prob=0.93, rank=1",
        xy=(0.98, 0.98), xycoords="axes fraction", ha="right", va="top",
        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
    )

    # ---- Panel (b): Hard token — M1 OK, M2 confused ----
    ax_b = fig.add_subplot(gs[0, 1])
    m1_labels = ["Token X", "Token Y", "Token A\n(correct)", "Token Z", "Others"]
    m1_hard = [0.30, 0.27, 0.22, 0.10, 0.11]
    m2_labels = ["Token P", "Token Q", "...98 more\ntokens...", "Token A\n(correct)", "Others"]
    m2_hard = [0.010, 0.009, 0.006, 0.004, 0.001]
    m2_hard_scaled = [v * 50 for v in m2_hard]  # scale for visibility

    ax_b_m1 = ax_b
    ax_b_m1.bar(
        np.arange(5) - w / 2, m1_hard, w,
        label=f"{m1_name} (actual scale)", color=COLORS["m1"], alpha=0.85,
    )
    ax_b_m2_twin = ax_b.twinx()
    ax_b_m2_twin.bar(
        np.arange(5) + w / 2, m2_hard, w,
        label=f"{m2_name} (actual scale)", color=COLORS["m2"], alpha=0.85,
    )
    ax_b_m1.set_ylabel(f"Probability ({m1_name})", fontsize=11, color=COLORS["m1"])
    ax_b_m2_twin.set_ylabel(f"Probability ({m2_name})", fontsize=11, color=COLORS["m2"])
    ax_b_m1.set_ylim(0, 0.5)
    ax_b_m2_twin.set_ylim(0, 0.015)
    ax_b_m1.set_xticks(np.arange(5))
    combined_labels = []
    for i in range(5):
        if m1_labels[i] == m2_labels[i]:
            combined_labels.append(m1_labels[i])
        else:
            combined_labels.append(f"{m1_labels[i]}\n({m2_labels[i]})")
    ax_b_m1.set_xticklabels(
        ["Top-1\n(M1: X, M2: P)", "Top-2\n(M1: Y, M2: Q)",
         "Top-3 or ...\n(M1: correct, M2: ...98 tokens)",
         "M2 rank-100\n(correct token)", "Others"],
        fontsize=8,
    )
    ax_b_m1.set_title(
        '(b) "Hard" Token — M1 Decent, M2 Lost',
        fontsize=13, fontweight="bold",
    )
    ax_b.annotate(
        f"{m1_name}: prob=0.22, rank=3\n{m2_name}: prob=0.004, rank≈100",
        xy=(0.98, 0.98), xycoords="axes fraction", ha="right", va="top",
        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
    )

    # ---- Panel (c): The averaging paradox ----
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.axis("off")

    text_content = (
        "The Averaging Paradox\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Suppose 80% of tokens are 'easy' and 20% are 'hard':\n\n"
        f"  {m1_name} (consistent):\n"
        "    Easy tokens: prob=0.70, rank=1\n"
        "    Hard tokens: prob=0.22, rank=3\n"
        "    → Avg prob = 0.80×0.70 + 0.20×0.22 = 0.604\n"
        "    → Avg rank = 0.80×1 + 0.20×3 = 1.4\n\n"
        f"  {m2_name} (polarized):\n"
        "    Easy tokens: prob=0.93, rank=1\n"
        "    Hard tokens: prob=0.004, rank=100\n"
        "    → Avg prob = 0.80×0.93 + 0.20×0.004 = 0.745\n"
        "    → Avg rank = 0.80×1 + 0.20×100 = 20.8\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  {m2_name} prob (0.745) > {m1_name} prob (0.604)  ✓\n"
        f"  {m2_name} rank (20.8) >> {m1_name} rank (1.4)  ✗\n\n"
        "Key: Prob is bounded [0,1] → robust to outliers.\n"
        "     Rank can be 1~150K → dominated by worst cases."
    )
    ax_c.text(
        0.05, 0.95, text_content, transform=ax_c.transAxes,
        fontsize=11, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor="#BDBDBD"),
    )

    # ---- Panel (d): Visual metaphor — two students ----
    ax_d = fig.add_subplot(gs[1, 1])
    categories = ["Easy Q\n(80%)", "Hard Q\n(20%)", "Overall\nAvg"]
    m1_conf = [0.70, 0.22, 0.604]
    m2_conf = [0.93, 0.004, 0.745]
    m1_rank_v = [1, 3, 1.4]
    m2_rank_v = [1, 100, 20.8]

    x = np.arange(3)
    w = 0.18

    ax_d.bar(x - 1.5 * w, m1_conf, w, label=f"{m1_name} prob", color=COLORS["m1"], alpha=0.9)
    ax_d.bar(x - 0.5 * w, m2_conf, w, label=f"{m2_name} prob", color=COLORS["m2"], alpha=0.9)

    ax_d2 = ax_d.twinx()
    ax_d2.bar(x + 0.5 * w, m1_rank_v, w, label=f"{m1_name} rank", color=COLORS["m1"],
              alpha=0.4, hatch="//")
    ax_d2.bar(x + 1.5 * w, m2_rank_v, w, label=f"{m2_name} rank", color=COLORS["m2"],
              alpha=0.4, hatch="//")

    ax_d.set_ylabel("Probability", fontsize=12, color="black")
    ax_d2.set_ylabel("Rank", fontsize=12, color="gray")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(categories, fontsize=11)
    ax_d.set_ylim(0, 1.1)
    ax_d2.set_yscale("log")
    ax_d2.set_ylim(0.5, 200)
    ax_d.set_title("(d) Summary: Prob vs Rank", fontsize=13, fontweight="bold")

    lines1, labels1 = ax_d.get_legend_handles_labels()
    lines2, labels2 = ax_d2.get_legend_handles_labels()
    ax_d.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    fig.suptitle(
        "Why Higher Probability ≠ Better Rank: A Conceptual Illustration",
        fontsize=16, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_token_prob_rank_joint(
    stats_m1: Dict, stats_m2: Dict, output_path: str,
    m1_name: str = "M1 (LST)", m2_name: str = "M2 (FFT)",
    max_points: int = 20000,
):
    """2D density / scatter of per-token (probability, rank) for both models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    for ax, stats, name, color in [
        (ax1, stats_m1, m1_name, COLORS["m1"]),
        (ax2, stats_m2, m2_name, COLORS["m2"]),
    ]:
        probs = stats["all_probs"]
        ranks = stats["all_ranks"]
        if len(probs) > max_points:
            idx = np.random.choice(len(probs), max_points, replace=False)
            probs = probs[idx]
            ranks = ranks[idx]

        ax.scatter(
            probs, ranks, alpha=0.08, s=3, c=color, edgecolors="none", rasterized=True,
        )
        ax.set_xlabel("Token Probability", fontsize=12)
        ax.set_yscale("log")
        ax.set_title(
            f"{name}\nmean_prob={stats['mean_prob']:.4f}  mean_rank={stats['mean_rank']:.1f}",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

    ax1.set_ylabel("Token Rank (log scale)", fontsize=12)
    fig.suptitle(
        "Per-Token Probability vs Rank", fontsize=15, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def print_paradox_report(examples: List[Dict], m1_name: str, m2_name: str):
    """Print a text report of the most paradoxical samples."""
    print(f"\n{'=' * 70}")
    print(f"TOP PARADOX EXAMPLES: {m2_name} prob > {m1_name} prob, "
          f"but {m2_name} rank >> {m1_name} rank")
    print(f"{'=' * 70}")

    for i, ex in enumerate(examples):
        print(f"\n--- Example {i + 1} ---")
        print(f"  Task: {ex['task']}")
        print(f"  Sample ID: {ex['sample_id']}")
        print(f"  Output (first 120 chars): {ex['output'][:120]}...")
        print(f"  {m1_name}: avg_prob={ex['m1_avg_prob']:.4f}, avg_rank={ex['m1_avg_rank']:.1f}")
        print(f"  {m2_name}: avg_prob={ex['m2_avg_prob']:.4f}, avg_rank={ex['m2_avg_rank']:.1f}")
        print(f"  Rank ratio ({m2_name}/{m1_name}): {ex['rank_ratio']:.1f}x")
        print(f"  Prob difference: +{ex['prob_diff']:.4f}")

        rec2 = ex["rec2"]
        if "token_ranks" in rec2 and "token_strs" in rec2:
            high_rank_tokens = [
                (rec2["token_strs"][j], rec2["token_ranks"][j], rec2["token_probs"][j])
                for j in range(len(rec2["token_ranks"]))
                if rec2["token_ranks"][j] > 50
            ]
            if high_rank_tokens:
                high_rank_tokens.sort(key=lambda x: x[1], reverse=True)
                print(f"  {m2_name} high-rank tokens (rank > 50):")
                for tok, rank, prob in high_rank_tokens[:8]:
                    print(f"    '{tok}' → rank={rank}, prob={prob:.4f}")


def save_paradox_report(
    examples: List[Dict], output_path: str, m1_name: str, m2_name: str,
):
    """Save detailed paradox report to text file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Probability-Rank Paradox Report\n")
        f.write(f"{'=' * 70}\n")
        f.write(f"Samples where {m2_name} has higher avg probability but worse avg rank "
                f"than {m1_name}.\n\n")

        for i, ex in enumerate(examples):
            f.write(f"{'─' * 70}\n")
            f.write(f"Example {i + 1}\n")
            f.write(f"  Task: {ex['task']}\n")
            f.write(f"  Sample ID: {ex['sample_id']}\n")
            f.write(f"  Output: {ex['output']}\n")
            f.write(f"  {m1_name}: avg_prob={ex['m1_avg_prob']:.4f}, "
                    f"avg_rank={ex['m1_avg_rank']:.1f}\n")
            f.write(f"  {m2_name}: avg_prob={ex['m2_avg_prob']:.4f}, "
                    f"avg_rank={ex['m2_avg_rank']:.1f}\n")
            f.write(f"  Rank ratio: {ex['rank_ratio']:.1f}x\n\n")

            rec1, rec2 = ex["rec1"], ex["rec2"]
            if "token_strs" in rec1 and "token_strs" in rec2:
                f.write(f"  {'Token':<20} │ {m1_name:>10} prob │ {m1_name:>10} rank │ "
                        f"{m2_name:>10} prob │ {m2_name:>10} rank │ ΔRank\n")
                f.write(f"  {'─' * 20}─┼─{'─' * 10}──────┼─{'─' * 10}──────┼─"
                        f"{'─' * 10}──────┼─{'─' * 10}──────┼──────\n")
                n = min(len(rec1["token_strs"]), len(rec2["token_strs"]))
                for j in range(n):
                    tok = repr(rec1["token_strs"][j])[:20]
                    p1 = rec1["token_probs"][j]
                    r1 = rec1["token_ranks"][j]
                    p2 = rec2["token_probs"][j]
                    r2 = rec2["token_ranks"][j]
                    delta = r2 - r1
                    marker = " <<<" if delta > 50 else ""
                    f.write(f"  {tok:<20} │ {p1:>10.4f}      │ {r1:>10}      │ "
                            f"{p2:>10.4f}      │ {r2:>10}      │ {delta:>+5}{marker}\n")
                f.write("\n")

    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize the probability-rank paradox"
    )
    parser.add_argument(
        "--m1_data", type=str, default=None,
        help="Path to M1 per-token JSONL (from collect_per_token_details.py)",
    )
    parser.add_argument(
        "--m2_data", type=str, default=None,
        help="Path to M2 per-token JSONL (from collect_per_token_details.py)",
    )
    parser.add_argument(
        "--m1_summary", type=str, default=None,
        help="Path to M1 summary.txt (for summary-only mode)",
    )
    parser.add_argument(
        "--m2_summary", type=str, default=None,
        help="Path to M2 summary.txt (for summary-only mode)",
    )
    parser.add_argument(
        "--m1_name", type=str, default="M1 (LST)",
    )
    parser.add_argument(
        "--m2_name", type=str, default="M2 (FFT)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="paradox_figures",
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=None,
        help="Limit analysis to specific tasks",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # # Always generate conceptual illustration
    # print("\n[1] Generating conceptual illustration...")
    # plot_conceptual_illustration(
    #     str(output_dir / "fig_conceptual_illustration.png"),
    #     args.m1_name, args.m2_name,
    # )

    # Summary-based plots
    if args.m1_summary and args.m2_summary:
        print("\n[2] Generating task-level comparison from summary files...")
        summary_m1 = parse_summary_file(args.m1_summary)
        summary_m2 = parse_summary_file(args.m2_summary)
        if args.tasks:
            summary_m1 = {k: v for k, v in summary_m1.items() if k in args.tasks}
            summary_m2 = {k: v for k, v in summary_m2.items() if k in args.tasks}
        plot_task_comparison(
            summary_m1, summary_m2,
            str(output_dir / "fig_task_comparison.png"),
            args.m1_name, args.m2_name,
        )

    # Per-token data analysis
    if args.m1_data and args.m2_data:
        print("\n[3] Loading per-token data...")
        data_m1 = load_per_token_data(args.m1_data)
        data_m2 = load_per_token_data(args.m2_data)

        if args.tasks:
            data_m1 = [r for r in data_m1 if r["task"] in args.tasks]
            data_m2 = [r for r in data_m2 if r["task"] in args.tasks]

        print(f"  M1 samples: {len(data_m1)}, M2 samples: {len(data_m2)}")

        # Task-level comparison from per-token data
        if not (args.m1_summary and args.m2_summary):
            groups_m1 = group_by_task(data_m1)
            groups_m2 = group_by_task(data_m2)
            summary_m1 = {
                t: {
                    "avg_prob": np.mean([r["avg_prob"] for r in recs]),
                    "avg_rank": np.mean([r["avg_rank"] for r in recs]),
                    "n_samples": len(recs),
                }
                for t, recs in groups_m1.items()
            }
            summary_m2 = {
                t: {
                    "avg_prob": np.mean([r["avg_prob"] for r in recs]),
                    "avg_rank": np.mean([r["avg_rank"] for r in recs]),
                    "n_samples": len(recs),
                }
                for t, recs in groups_m2.items()
            }
            print("\n[3a] Generating task-level comparison from per-token data...")
            plot_task_comparison(
                summary_m1, summary_m2,
                str(output_dir / "fig_task_comparison.png"),
                args.m1_name, args.m2_name,
            )

        # Rank distribution
        print("\n[4] Computing rank statistics...")
        stats_m1 = compute_rank_stats(data_m1)
        stats_m2 = compute_rank_stats(data_m2)

        print(f"\n  {args.m1_name} rank stats:")
        print(f"    Mean rank: {stats_m1['mean_rank']:.2f}")
        print(f"    Median rank: {stats_m1['median_rank']:.1f}")
        print(f"    P90 rank: {stats_m1['p90_rank']:.1f}")
        print(f"    P99 rank: {stats_m1['p99_rank']:.1f}")
        print(f"    Rank CDF: {stats_m1['rank_cdf']}")

        print(f"\n  {args.m2_name} rank stats:")
        print(f"    Mean rank: {stats_m2['mean_rank']:.2f}")
        print(f"    Median rank: {stats_m2['median_rank']:.1f}")
        print(f"    P90 rank: {stats_m2['p90_rank']:.1f}")
        print(f"    P99 rank: {stats_m2['p99_rank']:.1f}")
        print(f"    Rank CDF: {stats_m2['rank_cdf']}")

        print("\n[5] Generating rank distribution plot...")
        plot_rank_distribution(
            stats_m1, stats_m2,
            str(output_dir / "fig_rank_distribution.png"),
            args.m1_name, args.m2_name,
        )

        # Per-task rank distributions
        groups_m1 = group_by_task(data_m1)
        groups_m2 = group_by_task(data_m2)
        common_tasks = sorted(set(groups_m1.keys()) & set(groups_m2.keys()))
        for task in common_tasks:
            task_stats_m1 = compute_rank_stats(groups_m1[task])
            task_stats_m2 = compute_rank_stats(groups_m2[task])
            plot_rank_distribution(
                task_stats_m1, task_stats_m2,
                str(output_dir / f"fig_rank_distribution_{task}.png"),
                args.m1_name, args.m2_name, task_label=task,
            )

        # Token-level scatter
        print("\n[6] Generating per-token prob vs rank scatter...")
        plot_token_prob_rank_joint(
            stats_m1, stats_m2,
            str(output_dir / "fig_token_prob_rank_scatter.png"),
            args.m1_name, args.m2_name,
        )

        # Sample-level scatter
        print("\n[7] Matching samples and generating scatter...")
        pairs = match_samples(data_m1, data_m2)
        print(f"  Matched {len(pairs)} sample pairs")
        if pairs:
            plot_sample_scatter(
                pairs, str(output_dir / "fig_sample_scatter.png"),
                args.m1_name, args.m2_name,
            )

            # Find paradox examples
            print("\n[8] Finding paradox examples...")
            paradox_examples = find_paradox_examples(pairs, n=20)
            print(f"  Found {len(paradox_examples)} paradox examples")

            if paradox_examples:
                print_paradox_report(paradox_examples, args.m1_name, args.m2_name)
                save_paradox_report(
                    paradox_examples,
                    str(output_dir / "paradox_examples_report.txt"),
                    args.m1_name, args.m2_name,
                )

                # Case study for the top paradox example
                print("\n[9] Generating case study for top paradox example...")
                plot_case_study(
                    paradox_examples[0],
                    str(output_dir / "fig_case_study_top1.png"),
                    args.m1_name, args.m2_name,
                )
                if len(paradox_examples) > 1:
                    plot_case_study(
                        paradox_examples[1],
                        str(output_dir / "fig_case_study_top2.png"),
                        args.m1_name, args.m2_name,
                    )

    print(f"\n{'=' * 70}")
    print(f"All outputs saved to: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
