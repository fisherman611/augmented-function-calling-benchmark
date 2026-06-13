#!/usr/bin/env python
"""Summarize prompt lengths for generated prompt JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


EXPECTED_FILE_TYPES = ("interactive_agent", "search", "tool_call")
EXPECTED_PROMPT_TYPES = ("hallucination", "disambiguation")


@dataclass
class PromptLengthStats:
    source_split: str
    file_type: str
    prompt_type: str
    path: str
    prompts: int = 0
    invalid_json: int = 0
    missing_prompt: int = 0
    total_chars: int = 0
    total_bytes_utf8: int = 0
    min_chars: int = 0
    max_chars: int = 0
    mean_chars: float = 0.0
    median_chars: int = 0
    p90_chars: int = 0
    p95_chars: int = 0
    p99_chars: int = 0
    total_words: int = 0
    min_words: int = 0
    max_words: int = 0
    mean_words: float = 0.0
    median_words: int = 0
    p90_words: int = 0
    p95_words: int = 0
    p99_words: int = 0

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["mean_chars"] = round(self.mean_chars, 2)
        row["mean_words"] = round(self.mean_words, 2)
        return row


def iter_prompt_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("*/*/prompts.jsonl"))


def normalize_file_type(source_split: str) -> str:
    if source_split == "tool_calling":
        return "tool_call"
    return source_split


def percentile(sorted_values: list[int], percentile_value: float) -> int:
    if not sorted_values:
        return 0
    rank = math.ceil(percentile_value / 100 * len(sorted_values))
    index = min(max(rank - 1, 0), len(sorted_values) - 1)
    return sorted_values[index]


def summarize_file(path: Path, root: Path, limit: int | None, *, progress: bool) -> PromptLengthStats:
    relative = path.relative_to(root)
    source_split = relative.parts[0]
    prompt_type = relative.parts[1]
    stats = PromptLengthStats(
        source_split=source_split,
        file_type=normalize_file_type(source_split),
        prompt_type=prompt_type,
        path=str(relative).replace("\\", "/"),
    )

    char_lengths: list[int] = []
    word_lengths: list[int] = []
    progress_bar = None
    try:
        if progress and tqdm is not None:
            progress_bar = tqdm(
                desc=str(relative).replace("\\", "/"),
                unit="line",
                dynamic_ncols=True,
                file=sys.stdout,
                leave=False,
            )

        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if progress_bar is not None:
                    progress_bar.update(1)
                if limit is not None and stats.prompts >= limit:
                    break
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    stats.invalid_json += 1
                    continue
                prompt = record.get("prompt") if isinstance(record, dict) else None
                if not isinstance(prompt, str):
                    stats.missing_prompt += 1
                    continue

                prompt_chars = len(prompt)
                prompt_bytes = len(prompt.encode("utf-8"))
                prompt_words = len(prompt.split())
                char_lengths.append(prompt_chars)
                word_lengths.append(prompt_words)
                stats.prompts += 1
                stats.total_chars += prompt_chars
                stats.total_bytes_utf8 += prompt_bytes
                stats.total_words += prompt_words
    finally:
        if progress_bar is not None:
            progress_bar.close()

    if char_lengths:
        sorted_lengths = sorted(char_lengths)
        stats.min_chars = sorted_lengths[0]
        stats.max_chars = sorted_lengths[-1]
        stats.mean_chars = stats.total_chars / stats.prompts
        stats.median_chars = percentile(sorted_lengths, 50)
        stats.p90_chars = percentile(sorted_lengths, 90)
        stats.p95_chars = percentile(sorted_lengths, 95)
        stats.p99_chars = percentile(sorted_lengths, 99)
        sorted_word_lengths = sorted(word_lengths)
        stats.min_words = sorted_word_lengths[0]
        stats.max_words = sorted_word_lengths[-1]
        stats.mean_words = stats.total_words / stats.prompts
        stats.median_words = percentile(sorted_word_lengths, 50)
        stats.p90_words = percentile(sorted_word_lengths, 90)
        stats.p95_words = percentile(sorted_word_lengths, 95)
        stats.p99_words = percentile(sorted_word_lengths, 99)

    return stats


def add_totals(rows: list[PromptLengthStats]) -> list[PromptLengthStats]:
    output = list(rows)
    for prompt_type in EXPECTED_PROMPT_TYPES:
        group = [row for row in rows if row.prompt_type == prompt_type]
        if group:
            output.append(combine_rows(group, source_split="TOTAL", file_type="TOTAL", prompt_type=prompt_type))
    if rows:
        output.append(combine_rows(rows, source_split="TOTAL", file_type="TOTAL", prompt_type="TOTAL"))
    return output


def sort_key(row: PromptLengthStats) -> tuple[int, int]:
    file_index = EXPECTED_FILE_TYPES.index(row.file_type) if row.file_type in EXPECTED_FILE_TYPES else len(EXPECTED_FILE_TYPES)
    prompt_index = (
        EXPECTED_PROMPT_TYPES.index(row.prompt_type)
        if row.prompt_type in EXPECTED_PROMPT_TYPES
        else len(EXPECTED_PROMPT_TYPES)
    )
    return file_index, prompt_index


def combine_rows(
    rows: list[PromptLengthStats],
    *,
    source_split: str,
    file_type: str,
    prompt_type: str,
) -> PromptLengthStats:
    combined = PromptLengthStats(
        source_split=source_split,
        file_type=file_type,
        prompt_type=prompt_type,
        path="",
    )
    combined.prompts = sum(row.prompts for row in rows)
    combined.invalid_json = sum(row.invalid_json for row in rows)
    combined.missing_prompt = sum(row.missing_prompt for row in rows)
    combined.total_chars = sum(row.total_chars for row in rows)
    combined.total_bytes_utf8 = sum(row.total_bytes_utf8 for row in rows)
    combined.total_words = sum(row.total_words for row in rows)
    if combined.prompts:
        combined.min_chars = min(row.min_chars for row in rows if row.prompts)
        combined.max_chars = max(row.max_chars for row in rows if row.prompts)
        combined.mean_chars = combined.total_chars / combined.prompts
        combined.median_chars = max(row.median_chars for row in rows if row.prompts)
        combined.p90_chars = max(row.p90_chars for row in rows if row.prompts)
        combined.p95_chars = max(row.p95_chars for row in rows if row.prompts)
        combined.p99_chars = max(row.p99_chars for row in rows if row.prompts)
        combined.min_words = min(row.min_words for row in rows if row.prompts)
        combined.max_words = max(row.max_words for row in rows if row.prompts)
        combined.mean_words = combined.total_words / combined.prompts
        combined.median_words = max(row.median_words for row in rows if row.prompts)
        combined.p90_words = max(row.p90_words for row in rows if row.prompts)
        combined.p95_words = max(row.p95_words for row in rows if row.prompts)
        combined.p99_words = max(row.p99_words for row in rows if row.prompts)
    return combined


def write_outputs(output_path: Path, rows: list[PromptLengthStats]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_path.with_suffix(".json")
    csv_path = output_path.with_suffix(".csv")
    row_dicts = [row.to_row() for row in rows]

    json_path.write_text(json.dumps(row_dicts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_dicts[0].keys()))
        writer.writeheader()
        writer.writerows(row_dicts)


def print_table(rows: list[PromptLengthStats]) -> None:
    headers = ("file_type", "prompt_type", "prompts", "mean_words", "p95_words", "max_words", "mean_chars", "path")
    print("\t".join(headers))
    for row in rows:
        values = row.to_row()
        print("\t".join(str(values[name]) for name in headers))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize prompt lengths by file type and prompt type.")
    parser.add_argument(
        "--input",
        default="nemotron_sft_agentic_v2_prompt_jsonl",
        help="Root folder containing split/task prompt JSONL files.",
    )
    parser.add_argument(
        "--output",
        default="nemotron_sft_agentic_v2_prompt_jsonl/prompt_length_stats",
        help="Output path stem for .json and .csv stats.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max prompt rows to read per prompts.jsonl file.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.input).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input folder does not exist: {root}")

    prompt_files = list(iter_prompt_files(root))
    if not prompt_files:
        raise FileNotFoundError(f"No */*/prompts.jsonl files found under: {root}")

    use_progress = not args.no_progress
    if use_progress and tqdm is None:
        print("tqdm is not installed; running without progress bars.")
        use_progress = False

    rows = [summarize_file(path, root, args.limit, progress=use_progress) for path in prompt_files]
    rows = sorted(rows, key=sort_key)
    rows_with_totals = add_totals(rows)
    write_outputs(Path(args.output), rows_with_totals)
    print_table(rows_with_totals)
    print(f"Stats written to: {Path(args.output).with_suffix('.json')}")
    print(f"Stats written to: {Path(args.output).with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
