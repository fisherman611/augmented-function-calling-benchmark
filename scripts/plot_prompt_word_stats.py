#!/usr/bin/env python
"""Plot prompt word-length statistics from prompt_length_stats.json."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MATPLOTLIB = REPO_ROOT / ".deps" / "matplotlib"
if LOCAL_MATPLOTLIB.exists():
    sys.path.insert(0, str(LOCAL_MATPLOTLIB))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib"))


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return [row for row in rows if isinstance(row, dict)]


def format_label(row: dict[str, Any]) -> str:
    file_type = str(row["file_type"]).replace("_", " ")
    prompt_type = str(row["prompt_type"])
    prompt_short = "hallu" if prompt_type == "hallucination" else "disamb"
    return f"{file_type}\n{prompt_short}"


def plot_word_stats(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    detail_rows = [row for row in rows if row.get("file_type") != "TOTAL" and row.get("prompt_type") != "TOTAL"]
    if not detail_rows:
        raise ValueError("No detailed rows found to plot.")

    labels = [format_label(row) for row in detail_rows]
    mean_words = [float(row["mean_words"]) for row in detail_rows]
    p95_words = [float(row["p95_words"]) for row in detail_rows]
    max_words = [float(row["max_words"]) for row in detail_rows]
    counts = [int(row["prompts"]) for row in detail_rows]

    x = np.arange(len(labels))
    width = 0.28

    fig, ax = plt.subplots(figsize=(13, 7), dpi=180)
    mean_bars = ax.bar(x - width, mean_words, width, label="Mean words", color="#2563eb")
    p95_bars = ax.bar(x, p95_words, width, label="P95 words", color="#f97316")
    max_bars = ax.bar(x + width, max_words, width, label="Max words", color="#64748b")

    ax.set_title("Prompt Word Length by Source Split and Prompt Type", fontsize=16, pad=16)
    ax.set_ylabel("Words per prompt")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")

    for bars in (mean_bars, p95_bars, max_bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:,.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    for index, count in enumerate(counts):
        ax.annotate(
            f"n={count:,}",
            xy=(x[index], 0),
            xytext=(0, -36),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
        )

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot prompt word-length statistics.")
    parser.add_argument(
        "--stats",
        default="nemotron_sft_agentic_v2_prompt_jsonl/prompt_length_stats.json",
        help="Path to prompt_length_stats.json.",
    )
    parser.add_argument(
        "--output",
        default="nemotron_sft_agentic_v2_prompt_jsonl/prompt_word_length_stats.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_rows(Path(args.stats))
    plot_word_stats(rows, Path(args.output))
    print(f"Plot written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
