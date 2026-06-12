#!/usr/bin/env python
"""Filter ToolMind JSONL files to samples that contain tool calls.

The script streams JSONL files line by line, so it can handle large dataset
files without loading them fully into memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


@dataclass
class FileStats:
    input_file: str
    output_file: str
    total: int = 0
    kept: int = 0
    removed: int = 0
    invalid_json: int = 0

    @property
    def kept_percent(self) -> float:
        if self.total == 0:
            return 0.0
        return self.kept / self.total * 100.0

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["kept_percent"] = round(self.kept_percent, 4)
        return row


def has_tool_call(sample: dict[str, Any]) -> bool:
    """Return True when any conversation message has at least one tool call."""
    conversations = sample.get("conversations")
    if not isinstance(conversations, list):
        return False

    for message in conversations:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and len(tool_calls) > 0:
            return True
    return False


def iter_jsonl_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix != ".jsonl":
            raise ValueError(f"Input file must be .jsonl: {input_path}")
        yield input_path
        return

    yield from sorted(input_path.rglob("*.jsonl"))


def make_output_path(input_file: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_file():
        return output_root / input_file.name
    return output_root / input_file.relative_to(input_root)


def count_lines(input_file: Path) -> int:
    with input_file.open("rb") as handle:
        return sum(1 for _ in handle)


def filter_file(
    input_file: Path,
    output_file: Path,
    *,
    input_root: Path,
    dry_run: bool,
    skip_invalid: bool,
    progress: bool,
) -> FileStats:
    stats = FileStats(
        input_file=str(input_file.relative_to(input_root) if input_root.is_dir() else input_file.name),
        output_file=str(output_file),
    )

    writer = None
    if not dry_run:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        writer = output_file.open("w", encoding="utf-8", newline="\n")

    progress_bar = None
    try:
        if progress and tqdm is not None:
            progress_bar = tqdm(
                total=count_lines(input_file),
                desc=stats.input_file,
                unit="line",
                dynamic_ncols=True,
                file=sys.stdout,
                leave=False,
            )

        with input_file.open("r", encoding="utf-8") as reader:
            for line_number, line in enumerate(reader, start=1):
                if progress_bar is not None:
                    progress_bar.update(1)

                if not line.strip():
                    continue

                stats.total += 1
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as exc:
                    stats.invalid_json += 1
                    if skip_invalid:
                        continue
                    raise ValueError(f"Invalid JSON in {input_file}:{line_number}: {exc}") from exc

                if isinstance(sample, dict) and has_tool_call(sample):
                    stats.kept += 1
                    if writer is not None:
                        writer.write(line)
                else:
                    stats.removed += 1
    finally:
        if progress_bar is not None:
            progress_bar.close()
        if writer is not None:
            writer.close()

    return stats


def write_stats(output_root: Path, stats: list[FileStats]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [item.to_row() for item in stats]

    totals = FileStats(input_file="TOTAL", output_file="")
    totals.total = sum(item.total for item in stats)
    totals.kept = sum(item.kept for item in stats)
    totals.removed = sum(item.removed for item in stats)
    totals.invalid_json = sum(item.invalid_json for item in stats)
    rows.append(totals.to_row())

    stats_json = output_root / "filter_stats.json"
    stats_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    stats_csv = output_root / "filter_stats.csv"
    with stats_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(stats: list[FileStats]) -> None:
    total = sum(item.total for item in stats)
    kept = sum(item.kept for item in stats)
    removed = sum(item.removed for item in stats)
    invalid = sum(item.invalid_json for item in stats)
    percent = kept / total * 100.0 if total else 0.0

    print(f"Files processed: {len(stats)}")
    print(f"Total samples:   {total:,}")
    print(f"Kept samples:    {kept:,} ({percent:.2f}%)")
    print(f"Removed samples: {removed:,}")
    print(f"Invalid JSON:    {invalid:,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter JSONL dataset samples to only records containing assistant tool_calls."
    )
    parser.add_argument(
        "--input",
        default="toolmind",
        help="Input .jsonl file or directory to scan recursively. Default: toolmind",
    )
    parser.add_argument(
        "--output",
        default="toolmind_tool_calls_only",
        help="Output directory for filtered JSONL files and stats. Default: toolmind_tool_calls_only",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only compute stats; do not write filtered JSONL files.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip invalid JSON lines instead of failing.",
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_root = Path(args.output).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if output_root.exists() and not args.allow_existing_output and not args.dry_run:
        raise FileExistsError(
            f"Output directory already exists: {output_root}\n"
            "Use --allow-existing-output to write into it, or choose a new --output path."
        )

    files = list(iter_jsonl_files(input_path))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found under: {input_path}")

    use_progress = not args.no_progress
    if use_progress and tqdm is None:
        print("tqdm is not installed; running without progress bars.")
        use_progress = False

    stats: list[FileStats] = []
    file_iter = files
    if use_progress and tqdm is not None:
        file_iter = tqdm(files, desc="Files", unit="file", dynamic_ncols=True, file=sys.stdout)

    for input_file in file_iter:
        output_file = make_output_path(input_file, input_path, output_root)
        stats.append(
            filter_file(
                input_file,
                output_file,
                input_root=input_path,
                dry_run=args.dry_run,
                skip_invalid=args.skip_invalid,
                progress=use_progress,
            )
        )

    if not args.dry_run:
        write_stats(output_root, stats)

    print_summary(stats)
    if not args.dry_run:
        print(f"Stats written to: {output_root / 'filter_stats.json'}")
        print(f"Stats written to: {output_root / 'filter_stats.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
