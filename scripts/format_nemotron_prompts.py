#!/usr/bin/env python
"""Format filtered tool-call samples into single-field prompt JSONL.

The output is intended for prompt-only SFT/generation jobs. Each line contains
exactly one key:

{"prompt": "..."}
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from augment_toolmind_tasks import (  # noqa: E402
    build_actions,
    build_prompt,
    compact_source_payload,
    extract_tool_calls,
    load_prompt_templates,
    make_source_id,
    normalize_tools,
    select_anchor_call,
    select_disambiguation_target,
    select_mutation,
)


TASK_HALLUCINATION = "hallucination_missing_tool"
TASK_DISAMBIGUATION = "disambiguation_user"


@dataclass
class FileStats:
    input_file: str
    output_split: str
    read: int = 0
    hallucination_prompts: int = 0
    disambiguation_prompts: int = 0
    skipped_no_tool_call: int = 0
    invalid_json: int = 0

    @property
    def prompts_total(self) -> int:
        return self.hallucination_prompts + self.disambiguation_prompts

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["prompts_total"] = self.prompts_total
        return row


def iter_jsonl_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix != ".jsonl":
            raise ValueError(f"Input file must be .jsonl: {input_path}")
        yield input_path
        return

    yield from sorted(input_path.rglob("*.jsonl"))


def make_generic_task_id(source_file: str, source_line: int, task_type: str) -> str:
    safe = []
    normalized = source_file.replace("\\", "/")
    if normalized.endswith(".jsonl"):
        normalized = normalized[:-6]
    for char in normalized:
        if char.isalnum() or char in ("-", "_"):
            safe.append(char)
        else:
            safe.append("_")
    source_part = "_".join(part for part in "".join(safe).split("_") if part)
    return f"generated_{source_part}_{source_line}_{task_type}"


def output_split_name(input_file: Path) -> str:
    return input_file.stem


def normalize_source_sample(sample: dict[str, Any], *, source_file: str, source_line: int) -> dict[str, Any]:
    """Return a conversation-shaped copy with conversations mapped from messages."""
    normalized = copy.deepcopy(sample)
    if "conversations" not in normalized and isinstance(normalized.get("messages"), list):
        normalized["conversations"] = normalized["messages"]

    metadata = normalized.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = copy.deepcopy(metadata)
    metadata.update(
        {
            "source_file": source_file,
            "source_line": source_line,
        }
    )
    normalized["metadata"] = metadata
    return normalized


def compose_single_prompt(system_prompt: str, user_prompt: str) -> str:
    return f"<SYSTEM>\n{system_prompt}\n</SYSTEM>\n\n<USER>\n{user_prompt}\n</USER>"


def write_prompt(handle: Any, prompt: str) -> None:
    handle.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")


def format_file(
    input_file: Path,
    *,
    input_root: Path,
    hallucination_handle: Any,
    disambiguation_handle: Any,
    prompt_templates: Any,
    seed: int,
    max_source_chars: int,
    limit_remaining: int | None,
    skip_invalid: bool,
    progress: bool,
) -> tuple[FileStats, int | None]:
    source_file = str(input_file.relative_to(input_root) if input_root.is_dir() else input_file.name).replace("\\", "/")
    stats = FileStats(input_file=source_file, output_split=output_split_name(input_file))

    progress_bar = None
    try:
        if progress and tqdm is not None:
            progress_bar = tqdm(desc=source_file, unit="line", dynamic_ncols=True, file=sys.stdout, leave=False)

        with input_file.open("r", encoding="utf-8") as reader:
            for line_number, line in enumerate(reader, start=1):
                if progress_bar is not None:
                    progress_bar.update(1)

                if limit_remaining is not None and limit_remaining <= 0:
                    break
                if not line.strip():
                    continue

                stats.read += 1
                if limit_remaining is not None:
                    limit_remaining -= 1

                try:
                    raw_sample = json.loads(line)
                except json.JSONDecodeError as exc:
                    stats.invalid_json += 1
                    if skip_invalid:
                        continue
                    raise ValueError(f"Invalid JSON in {input_file}:{line_number}: {exc}") from exc

                if not isinstance(raw_sample, dict):
                    stats.skipped_no_tool_call += 1
                    continue

                sample = normalize_source_sample(raw_sample, source_file=source_file, source_line=line_number)
                tool_calls = extract_tool_calls(sample)
                anchor_call = select_anchor_call(tool_calls)
                if anchor_call is None:
                    stats.skipped_no_tool_call += 1
                    continue

                source_id = make_source_id(source_file, line_number)
                source_payload = compact_source_payload(sample, max_source_chars)
                actions = build_actions(tool_calls)
                schemas = normalize_tools(sample.get("tools", []))

                mutation = select_mutation(
                    source_id,
                    seed,
                    sample,
                    anchor_call,
                    schemas.get(anchor_call["name"]),
                )
                hallucination_prompt = build_prompt(
                    TASK_HALLUCINATION,
                    prompt_templates=prompt_templates,
                    task_id=make_generic_task_id(source_file, line_number, TASK_HALLUCINATION),
                    source_id=source_id,
                    source_payload=source_payload,
                    actions=actions,
                    anchor_call=anchor_call,
                    mutation=mutation,
                    disambiguation_target=None,
                    validation_feedback=None,
                )
                write_prompt(
                    hallucination_handle,
                    compose_single_prompt(prompt_templates.system, hallucination_prompt),
                )
                stats.hallucination_prompts += 1

                disambiguation_target = select_disambiguation_target(source_id, seed, tool_calls, schemas)
                disambiguation_prompt = build_prompt(
                    TASK_DISAMBIGUATION,
                    prompt_templates=prompt_templates,
                    task_id=make_generic_task_id(source_file, line_number, TASK_DISAMBIGUATION),
                    source_id=source_id,
                    source_payload=source_payload,
                    actions=actions,
                    anchor_call=anchor_call,
                    mutation=None,
                    disambiguation_target=disambiguation_target,
                    validation_feedback=None,
                )
                write_prompt(
                    disambiguation_handle,
                    compose_single_prompt(prompt_templates.system, disambiguation_prompt),
                )
                stats.disambiguation_prompts += 1

    finally:
        if progress_bar is not None:
            progress_bar.close()

    return stats, limit_remaining


def write_stats(output_root: Path, stats: list[FileStats]) -> None:
    rows = [item.to_row() for item in stats]
    totals = FileStats(input_file="TOTAL", output_split="")
    totals.read = sum(item.read for item in stats)
    totals.hallucination_prompts = sum(item.hallucination_prompts for item in stats)
    totals.disambiguation_prompts = sum(item.disambiguation_prompts for item in stats)
    totals.skipped_no_tool_call = sum(item.skipped_no_tool_call for item in stats)
    totals.invalid_json = sum(item.invalid_json for item in stats)
    rows.append(totals.to_row())

    (output_root / "format_stats.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def print_summary(stats: list[FileStats]) -> None:
    read = sum(item.read for item in stats)
    hallucination = sum(item.hallucination_prompts for item in stats)
    disambiguation = sum(item.disambiguation_prompts for item in stats)
    skipped = sum(item.skipped_no_tool_call for item in stats)
    invalid = sum(item.invalid_json for item in stats)

    print(f"Files processed:          {len(stats)}")
    print(f"Samples read:             {read:,}")
    print(f"Hallucination prompts:    {hallucination:,}")
    print(f"Disambiguation prompts:   {disambiguation:,}")
    print(f"Skipped no tool call:     {skipped:,}")
    print(f"Invalid JSON:             {invalid:,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Format filtered tool-call samples into {\"prompt\": ...} JSONL files."
    )
    parser.add_argument(
        "--input",
        default="nemotron_sft_agentic_v2_tool_calls_only",
        help="Input filtered .jsonl file or directory.",
    )
    parser.add_argument(
        "--output",
        default="nemotron_sft_agentic_v2_prompt_jsonl",
        help="Output directory containing one top-level folder per original input split.",
    )
    parser.add_argument("--prompts-dir", default="prompts", help="Directory containing system/task prompt files.")
    parser.add_argument("--seed", type=int, default=13, help="Seed for deterministic mutation/target selection.")
    parser.add_argument(
        "--max-source-chars",
        type=int,
        default=60000,
        help="Maximum source sample characters included in each rendered prompt.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum source samples to process across all files.")
    parser.add_argument("--limit-per-file", type=int, default=None, help="Maximum source samples to process per input file.")
    parser.add_argument("--skip-invalid", action="store_true", help="Skip invalid JSON lines instead of failing.")
    parser.add_argument("--allow-existing-output", action="store_true", help="Allow overwriting prompt JSONL files.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    prompts_dir = Path(args.prompts_dir).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if output_root.exists() and not args.allow_existing_output:
        raise FileExistsError(
            f"Output directory already exists: {output_root}\n"
            "Use --allow-existing-output to overwrite prompt JSONL files, or choose a new --output path."
        )

    files = list(iter_jsonl_files(input_path))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found under: {input_path}")

    prompt_templates = load_prompt_templates(prompts_dir)

    use_progress = not args.no_progress
    if use_progress and tqdm is None:
        print("tqdm is not installed; running without progress bars.")
        use_progress = False

    stats: list[FileStats] = []
    limit_remaining = args.limit
    written_outputs: list[tuple[Path, Path]] = []
    file_iter = files
    if use_progress and tqdm is not None:
        file_iter = tqdm(files, desc="Files", unit="file", dynamic_ncols=True, file=sys.stdout)

    for input_file in file_iter:
        if limit_remaining is not None and limit_remaining <= 0:
            break

        split_dir = output_root / output_split_name(input_file)
        hallucination_dir = split_dir / "hallucination"
        disambiguation_dir = split_dir / "disambiguation"
        hallucination_dir.mkdir(parents=True, exist_ok=True)
        disambiguation_dir.mkdir(parents=True, exist_ok=True)

        file_limit_remaining = args.limit_per_file
        if limit_remaining is not None:
            file_limit_remaining = limit_remaining if file_limit_remaining is None else min(file_limit_remaining, limit_remaining)

        hallucination_path = hallucination_dir / "prompts.jsonl"
        disambiguation_path = disambiguation_dir / "prompts.jsonl"
        with hallucination_path.open("w", encoding="utf-8", newline="\n") as hallucination_handle, disambiguation_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as disambiguation_handle:
            file_stats, file_limit_remaining = format_file(
                input_file,
                input_root=input_path,
                hallucination_handle=hallucination_handle,
                disambiguation_handle=disambiguation_handle,
                prompt_templates=prompt_templates,
                seed=args.seed,
                max_source_chars=args.max_source_chars,
                limit_remaining=file_limit_remaining,
                skip_invalid=args.skip_invalid,
                progress=use_progress,
            )
        if args.limit is not None:
            if args.limit_per_file is None:
                limit_remaining = file_limit_remaining
            else:
                limit_remaining -= file_stats.read
        stats.append(file_stats)
        written_outputs.append((hallucination_path, disambiguation_path))

    write_stats(output_root, stats)
    print_summary(stats)
    for hallucination_path, disambiguation_path in written_outputs:
        print(f"Hallucination prompts written to:  {hallucination_path}")
        print(f"Disambiguation prompts written to: {disambiguation_path}")
    print(f"Stats written to:                 {output_root / 'format_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
