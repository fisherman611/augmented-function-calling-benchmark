#!/usr/bin/env python
"""Generate Augmented ToolMind records with LLM-authored task variants.

For every ToolMind JSONL sample, this script attempts to generate three
Augmented ToolMind task records:

* base
* hallucination_missing_tool
* disambiguation_user

The script streams input JSONL files and writes output JSONL incrementally.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


TASK_TYPES = ("base", "hallucination_missing_tool", "disambiguation_user")
NVIDIA_NIM_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_PERSONA = (
    "You are a practical user asking an assistant to complete the requested task. "
    "You value concise, correct tool use and clear follow-up questions when needed."
)
SYSTEM_PROMPT = """You generate Augmented ToolMind benchmark records.

Return exactly one valid JSON object and no surrounding prose.
Do not use markdown fences.
Use Augmented ToolMind naming only; do not introduce unrelated benchmark names.
Use the requested task_id and task_type exactly.
The generated_trace.messages field must use OpenAI-style messages:
- user/assistant messages have role and content.
- assistant tool-call messages may include tool_calls.
- tool result messages use role "tool", name when known, tool_call_id when known, and content.

The JSON object must include these top-level fields:
task_id, task_type, persona, instruction, context_init_config, actions,
removed_part, disambiguation_element_user, disambiguation_element_note,
generated_trace, metadata.

context_init_config must be a JSON-encoded string. Use "{}" unless a harmless
context is directly supported by the source sample.
actions must be a JSON-encoded string containing a list of action dictionaries.
metadata can be an object; source metadata will be overwritten by the script.
"""


@dataclass
class FileStats:
    input_file: str
    output_file: str
    input: int = 0
    base_written: int = 0
    hallucination_written: int = 0
    disambiguation_written: int = 0
    failed_generation: int = 0
    failed_validation: int = 0
    skipped: int = 0

    @property
    def output_total(self) -> int:
        return self.base_written + self.hallucination_written + self.disambiguation_written

    @property
    def target_total(self) -> int:
        return self.input * 3

    @property
    def output_percent(self) -> float:
        if self.target_total == 0:
            return 0.0
        return self.output_total / self.target_total * 100.0

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["output_total"] = self.output_total
        row["target_total"] = self.target_total
        row["output_percent"] = round(self.output_percent, 4)
        return row


@dataclass
class GenerationConfig:
    provider: str
    model: str
    api_key_env: str
    base_url: str | None
    temperature: float
    max_output_tokens: int
    retry_count: int
    seed: int
    max_source_chars: int
    disable_response_format: bool


class ValidationError(Exception):
    pass


class GenerationError(Exception):
    pass


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


def load_env_file(env_file: Path) -> None:
    """Load KEY=VALUE pairs from a .env file without overriding the shell env."""
    if not env_file.exists():
        return

    with env_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or key in os.environ:
                continue

            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            os.environ[key] = value


def safe_id_part(value: str) -> str:
    normalized = value.replace("\\", "/")
    if normalized.endswith(".jsonl"):
        normalized = normalized[:-6]
    chars = []
    for char in normalized:
        if char.isalnum() or char in ("-", "_"):
            chars.append(char)
        else:
            chars.append("_")
    return "_".join(part for part in "".join(chars).split("_") if part)


def make_task_id(source_file: str, source_line: int, task_type: str) -> str:
    return f"augmented_toolmind_{safe_id_part(source_file)}_{source_line}_{task_type}"


def make_source_id(source_file: str, source_line: int) -> str:
    return f"{source_file}:{source_line}"


def deterministic_index(seed: int, source_id: str, purpose: str, choices: int) -> int:
    digest = hashlib.sha256(f"{seed}:{source_id}:{purpose}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % choices


def coerce_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        if not value.strip():
            return {}
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"__raw_arguments__": value}
        return parsed if isinstance(parsed, dict) else {"__raw_arguments__": parsed}
    if value is None:
        return {}
    return {"__raw_arguments__": value}


def unwrap_function(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function", tool)
    seen = 0
    while isinstance(function, dict) and isinstance(function.get("function"), dict) and seen < 5:
        inner = function["function"]
        if "name" in inner or "parameters" in inner or "arguments" in inner:
            function = inner
        else:
            break
        seen += 1
    return function if isinstance(function, dict) else {}


def normalize_tool_schema(tool: dict[str, Any]) -> dict[str, Any] | None:
    function = unwrap_function(tool)
    name = function.get("name")
    if not isinstance(name, str) or not name:
        return None

    parameters = function.get("parameters")
    if not isinstance(parameters, dict):
        arguments_schema = function.get("arguments")
        parameters = arguments_schema if isinstance(arguments_schema, dict) else {}

    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        properties = {}

    required = parameters.get("required", function.get("required", []))
    if not isinstance(required, list):
        required = []

    return {
        "name": name,
        "description": function.get("description", ""),
        "parameters": parameters,
        "properties": properties,
        "required": [item for item in required if isinstance(item, str)],
        "raw": tool,
    }


def normalize_tools(tools: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(tools, list):
        return {}
    schemas: dict[str, dict[str, Any]] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        schema = normalize_tool_schema(tool)
        if schema is not None:
            schemas[schema["name"]] = schema
    return schemas


def extract_tool_calls(sample: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    conversations = sample.get("conversations")
    if not isinstance(conversations, list):
        return calls

    for message_index, message in enumerate(conversations):
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call_index, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function", {})
            if not isinstance(function, dict):
                function = {}
            name = function.get("name", tool_call.get("name"))
            if not isinstance(name, str) or not name:
                continue
            calls.append(
                {
                    "message_index": message_index,
                    "call_index": call_index,
                    "name": name,
                    "arguments": coerce_arguments(function.get("arguments", tool_call.get("arguments"))),
                    "raw": tool_call,
                }
            )
    return calls


def select_anchor_call(tool_calls: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not tool_calls:
        return None
    first_message_index = tool_calls[0]["message_index"]
    first_turn_calls = [call for call in tool_calls if call["message_index"] == first_message_index]
    for call in first_turn_calls:
        if call["arguments"]:
            return call
    return first_turn_calls[0]


def previous_user_index(conversations: list[Any], message_index: int) -> int | None:
    for index in range(message_index - 1, -1, -1):
        message = conversations[index]
        if isinstance(message, dict) and message.get("role") == "user":
            return index
    return None


def first_tool_result_after(conversations: list[Any], message_index: int) -> dict[str, Any] | None:
    for index in range(message_index + 1, len(conversations)):
        message = conversations[index]
        if isinstance(message, dict) and message.get("role") == "tool":
            return {"message_index": index, "message": message}
    return None


def choose_argument(schema: dict[str, Any] | None, arguments: dict[str, Any]) -> str | None:
    if not arguments:
        return None
    required = schema.get("required", []) if schema else []
    for name in required:
        if name in arguments:
            return name
    for name, value in arguments.items():
        if name.startswith("__"):
            continue
        if value not in (None, "", [], {}):
            return name
    return next(iter(arguments))


def build_actions(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions = []
    for index, call in enumerate(tool_calls):
        actions.append(
            {
                "name": call["name"],
                "kwargs": call["arguments"],
                "index": index,
                "dependent_on_action_index": None,
            }
        )
    return actions


def truncate_value(value: Any, max_string_chars: int = 2000) -> Any:
    if isinstance(value, str):
        if len(value) <= max_string_chars:
            return value
        return value[:max_string_chars] + f"... [truncated {len(value) - max_string_chars} chars]"
    if isinstance(value, list):
        return [truncate_value(item, max_string_chars) for item in value]
    if isinstance(value, dict):
        return {key: truncate_value(item, max_string_chars) for key, item in value.items()}
    return value


def compact_source_payload(sample: dict[str, Any], max_chars: int) -> dict[str, Any]:
    payload = {
        "conversations": sample.get("conversations", []),
        "tools": sample.get("tools", []),
    }
    text = json.dumps(payload, ensure_ascii=False)
    if len(text) <= max_chars:
        return payload

    compact = truncate_value(payload, max_string_chars=1000)
    text = json.dumps(compact, ensure_ascii=False)
    if len(text) <= max_chars:
        return compact

    return {
        "conversations": truncate_value(sample.get("conversations", [])[:8], max_string_chars=700),
        "tools": truncate_value(sample.get("tools", [])[:20], max_string_chars=700),
        "truncation_note": f"Source payload exceeded {max_chars} characters and was truncated for prompting.",
    }


def select_mutation(
    source_id: str,
    seed: int,
    sample: dict[str, Any],
    anchor_call: dict[str, Any],
    schema: dict[str, Any] | None,
) -> dict[str, Any]:
    conversations = sample.get("conversations", [])
    if not isinstance(conversations, list):
        conversations = []

    candidates: list[dict[str, Any]] = [
        {
            "mutation_type": "remove_tool",
            "removed_tool": anchor_call["name"],
            "removed_part": [anchor_call["name"]],
            "instruction": "The needed tool is unavailable. The assistant must not call it.",
        }
    ]

    argument_name = choose_argument(schema, anchor_call["arguments"])
    if argument_name is not None:
        candidates.append(
            {
                "mutation_type": "obscure_required_argument",
                "tool": anchor_call["name"],
                "argument_name": argument_name,
                "argument_value": anchor_call["arguments"].get(argument_name),
                "removed_part": [f"{anchor_call['name']}.{argument_name}"],
                "instruction": "A needed argument is missing or ambiguous. The assistant must ask for it or state that the request is underspecified.",
            }
        )

    tool_result = first_tool_result_after(conversations, anchor_call["message_index"])
    if tool_result is not None:
        content = tool_result["message"].get("content", "")
        candidates.append(
            {
                "mutation_type": "remove_tool_result",
                "tool": anchor_call["name"],
                "removed_result_message_index": tool_result["message_index"],
                "removed_result_excerpt": str(content)[:500],
                "removed_part": [f"tool_result_after_{anchor_call['name']}"],
                "instruction": "The needed tool result is unavailable. The assistant must not fabricate the missing result.",
            }
        )

    return candidates[deterministic_index(seed, source_id, "hallucination", len(candidates))]


def select_disambiguation_target(
    source_id: str,
    seed: int,
    tool_calls: list[dict[str, Any]],
    schemas: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for call in tool_calls:
        schema = schemas.get(call["name"])
        argument_name = choose_argument(schema, call["arguments"])
        if argument_name is not None:
            candidates.append(
                {
                    "tool": call["name"],
                    "argument_name": argument_name,
                    "argument_value": call["arguments"].get(argument_name),
                    "message_index": call["message_index"],
                    "call_index": call["call_index"],
                }
            )

    if not candidates:
        call = tool_calls[0]
        return {
            "tool": call["name"],
            "argument_name": None,
            "argument_value": None,
            "message_index": call["message_index"],
            "call_index": call["call_index"],
        }

    return candidates[deterministic_index(seed, source_id, "disambiguation", len(candidates))]


def build_prompt(
    task_type: str,
    *,
    task_id: str,
    source_id: str,
    source_payload: dict[str, Any],
    actions: list[dict[str, Any]],
    anchor_call: dict[str, Any],
    mutation: dict[str, Any] | None,
    disambiguation_target: dict[str, Any] | None,
    validation_feedback: str | None,
) -> str:
    instructions: dict[str, str] = {
        "base": """Create a base Augmented ToolMind record.
Preserve the source task intent, tool names, tool call order, tool arguments,
and final-answer behavior. Do not invent new tools. The generated_trace must
contain at least one assistant tool call.""",
        "hallucination_missing_tool": """Create a hallucination_missing_tool Augmented ToolMind record.
Use the mutation target to make the original task impossible or underspecified.
The assistant must not fabricate unavailable data and must not call tools that
the mutation removes. The assistant should clearly explain what is missing and,
when useful, ask the user for a valid next step.""",
        "disambiguation_user": """Create a disambiguation_user Augmented ToolMind record.
Rewrite the instruction so the selected argument is missing or ambiguous.
The first assistant response must ask a clarification question and must not
include tool_calls. You may continue the trace after the user supplies the
missing detail if it stays consistent with the source sample.""",
    }

    payload = {
        "task_id": task_id,
        "task_type": task_type,
        "source_id": source_id,
        "source_sample": source_payload,
        "extracted_actions": actions,
        "anchor_tool_call": anchor_call,
        "mutation": mutation,
        "disambiguation_target": disambiguation_target,
        "required_output_shape": {
            "task_id": task_id,
            "task_type": task_type,
            "persona": DEFAULT_PERSONA,
            "instruction": "user-facing task instruction",
            "context_init_config": "{}",
            "actions": json.dumps(actions, ensure_ascii=False),
            "removed_part": None,
            "disambiguation_element_user": None,
            "disambiguation_element_note": None,
            "generated_trace": {
                "trajectory_id": f"traj_{task_id}",
                "task_id": task_id,
                "messages": [],
            },
            "metadata": {},
        },
    }
    if validation_feedback:
        payload["previous_validation_error"] = validation_feedback

    return instructions[task_type] + "\n\nInput JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


class LLMGenerator:
    def __init__(self, config: GenerationConfig):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("The openai package is required. Install it with `python -m pip install openai`.") from exc

        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{config.api_key_env} is not set. Set it in your shell environment "
                "or in the file passed via --env-file."
            )

        kwargs: dict[str, Any] = {"api_key": api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self.client = OpenAI(**kwargs)
        self.config = config

    def complete_json(self, prompt: str) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
        }
        if not self.config.disable_response_format:
            request["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**request)
        content = response.choices[0].message.content
        if not content:
            raise GenerationError("LLM returned empty content.")
        return parse_json_object(content)


def parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON must be an object.")
    return parsed


def has_generated_tool_call(record: dict[str, Any]) -> bool:
    messages = record.get("generated_trace", {}).get("messages", [])
    if not isinstance(messages, list):
        return False
    for message in messages:
        if isinstance(message, dict) and isinstance(message.get("tool_calls"), list) and message["tool_calls"]:
            return True
    return False


def generated_tool_names(record: dict[str, Any]) -> list[str]:
    names: list[str] = []
    messages = record.get("generated_trace", {}).get("messages", [])
    if not isinstance(messages, list):
        return names
    for message in messages:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function", {})
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                names.append(function["name"])
    return names


def first_assistant_message(record: dict[str, Any]) -> dict[str, Any] | None:
    messages = record.get("generated_trace", {}).get("messages", [])
    if not isinstance(messages, list):
        return None
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            return message
    return None


def validate_record(record: dict[str, Any], task_type: str, mutation: dict[str, Any] | None) -> None:
    required = [
        "task_id",
        "task_type",
        "persona",
        "instruction",
        "context_init_config",
        "actions",
        "removed_part",
        "disambiguation_element_user",
        "disambiguation_element_note",
        "generated_trace",
        "metadata",
    ]
    for key in required:
        if key not in record:
            raise ValidationError(f"Missing top-level field: {key}")

    if record.get("task_type") != task_type:
        raise ValidationError(f"Unexpected task_type: {record.get('task_type')}")

    trace = record.get("generated_trace")
    if not isinstance(trace, dict) or not isinstance(trace.get("messages"), list):
        raise ValidationError("generated_trace.messages must be a list.")

    if not isinstance(record.get("actions"), str):
        raise ValidationError("actions must be a JSON-encoded string.")
    try:
        actions_value = json.loads(record["actions"])
    except json.JSONDecodeError as exc:
        raise ValidationError(f"actions is not valid JSON: {exc}") from exc
    if not isinstance(actions_value, list):
        raise ValidationError("actions JSON must decode to a list.")

    if not isinstance(record.get("context_init_config"), str):
        raise ValidationError("context_init_config must be a JSON-encoded string.")
    try:
        context_value = json.loads(record["context_init_config"])
    except json.JSONDecodeError as exc:
        raise ValidationError(f"context_init_config is not valid JSON: {exc}") from exc
    if not isinstance(context_value, dict):
        raise ValidationError("context_init_config JSON must decode to an object.")

    if task_type == "base" and not has_generated_tool_call(record):
        raise ValidationError("base record must contain at least one tool call.")

    if task_type == "hallucination_missing_tool" and mutation:
        if mutation.get("mutation_type") == "remove_tool":
            removed = mutation.get("removed_tool")
            if removed in generated_tool_names(record):
                raise ValidationError(f"hallucination record called removed tool: {removed}")
        if mutation.get("mutation_type") == "remove_tool_result":
            excerpt = mutation.get("removed_result_excerpt")
            if excerpt:
                serialized_trace = json.dumps(trace, ensure_ascii=False)
                if str(excerpt)[:120] in serialized_trace:
                    raise ValidationError("hallucination record reused removed tool result content.")

    if task_type == "disambiguation_user":
        assistant = first_assistant_message(record)
        if assistant is None:
            raise ValidationError("disambiguation record must include an assistant clarification.")
        if assistant.get("tool_calls"):
            raise ValidationError("first disambiguation assistant message must not include tool_calls.")
        content = str(assistant.get("content", "")).strip()
        if "?" not in content and not any(word in content.lower() for word in ("clarify", "provide", "which", "what", "need")):
            raise ValidationError("first disambiguation assistant message does not look like a clarification.")
        if not record.get("disambiguation_element_user") or not record.get("disambiguation_element_note"):
            raise ValidationError("disambiguation fields must be filled.")


def normalize_record(
    record: dict[str, Any],
    *,
    task_id: str,
    task_type: str,
    source_file: str,
    source_line: int,
    config: GenerationConfig,
    mutation: dict[str, Any] | None,
    original_sample: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized = copy.deepcopy(record)
    normalized["task_id"] = task_id
    normalized["task_type"] = task_type
    normalized.setdefault("persona", DEFAULT_PERSONA)
    normalized.setdefault("instruction", "")
    normalized.setdefault("context_init_config", "{}")
    normalized.setdefault("actions", "[]")
    normalized.setdefault("removed_part", None)
    normalized.setdefault("disambiguation_element_user", None)
    normalized.setdefault("disambiguation_element_note", None)

    trace = normalized.setdefault("generated_trace", {})
    if not isinstance(trace, dict):
        trace = {}
        normalized["generated_trace"] = trace
    trace["trajectory_id"] = f"traj_{task_id}"
    trace["task_id"] = task_id
    trace.setdefault("messages", [])

    metadata = normalized.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        normalized["metadata"] = metadata
    metadata.update(
        {
            "source_file": source_file,
            "source_line": source_line,
            "provider": config.provider,
            "model": config.model,
            "mutation": mutation or {},
        }
    )
    if original_sample is not None:
        metadata["original_sample"] = original_sample

    return normalized


def generate_task_record(
    generator: LLMGenerator,
    *,
    task_type: str,
    task_id: str,
    source_id: str,
    source_file: str,
    source_line: int,
    source_payload: dict[str, Any],
    original_sample: dict[str, Any],
    actions: list[dict[str, Any]],
    anchor_call: dict[str, Any],
    mutation: dict[str, Any] | None,
    disambiguation_target: dict[str, Any] | None,
    config: GenerationConfig,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    validation_feedback: str | None = None
    last_generation_error: str | None = None
    last_validation_error: str | None = None

    for attempt in range(1, config.retry_count + 1):
        try:
            prompt = build_prompt(
                task_type,
                task_id=task_id,
                source_id=source_id,
                source_payload=source_payload,
                actions=actions,
                anchor_call=anchor_call,
                mutation=mutation,
                disambiguation_target=disambiguation_target,
                validation_feedback=validation_feedback,
            )
            record = generator.complete_json(prompt)
        except Exception as exc:
            last_generation_error = f"attempt {attempt}: {type(exc).__name__}: {exc}"
            time.sleep(min(2**attempt, 8))
            continue

        try:
            normalized = normalize_record(
                record,
                task_id=task_id,
                task_type=task_type,
                source_file=source_file,
                source_line=source_line,
                config=config,
                mutation=mutation,
                original_sample=original_sample if task_type != "base" else None,
            )
            validate_record(normalized, task_type, mutation)
            return normalized, None, None
        except Exception as exc:
            last_validation_error = f"attempt {attempt}: {type(exc).__name__}: {exc}"
            validation_feedback = last_validation_error
            time.sleep(min(2**attempt, 8))

    return None, last_generation_error, last_validation_error


def write_stats(output_root: Path, stats: list[FileStats]) -> None:
    rows = [item.to_row() for item in stats]

    totals = FileStats(input_file="TOTAL", output_file="")
    totals.input = sum(item.input for item in stats)
    totals.base_written = sum(item.base_written for item in stats)
    totals.hallucination_written = sum(item.hallucination_written for item in stats)
    totals.disambiguation_written = sum(item.disambiguation_written for item in stats)
    totals.failed_generation = sum(item.failed_generation for item in stats)
    totals.failed_validation = sum(item.failed_validation for item in stats)
    totals.skipped = sum(item.skipped for item in stats)
    rows.append(totals.to_row())

    stats_json = output_root / "augmentation_stats.json"
    stats_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    stats_csv = output_root / "augmentation_stats.csv"
    with stats_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(stats: list[FileStats]) -> None:
    total_input = sum(item.input for item in stats)
    total_output = sum(item.output_total for item in stats)
    target = total_input * 3
    percent = total_output / target * 100.0 if target else 0.0
    print(f"Files processed:        {len(stats)}")
    print(f"Input samples:          {total_input:,}")
    print(f"Output records:         {total_output:,} / {target:,} ({percent:.2f}%)")
    print(f"Base written:           {sum(item.base_written for item in stats):,}")
    print(f"Hallucination written:  {sum(item.hallucination_written for item in stats):,}")
    print(f"Disambiguation written: {sum(item.disambiguation_written for item in stats):,}")
    print(f"Failed generation:      {sum(item.failed_generation for item in stats):,}")
    print(f"Failed validation:      {sum(item.failed_validation for item in stats):,}")
    print(f"Skipped tasks:          {sum(item.skipped for item in stats):,}")


def write_error(
    error_handle: Any,
    *,
    source_file: str,
    source_line: int,
    task_type: str,
    generation_error: str | None,
    validation_error: str | None,
    mutation: dict[str, Any] | None,
) -> None:
    payload = {
        "source_file": source_file,
        "source_line": source_line,
        "source_id": make_source_id(source_file, source_line),
        "task_type": task_type,
        "generation_error": generation_error,
        "validation_error": validation_error,
        "mutation": mutation or {},
    }
    error_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    error_handle.flush()


def process_file(
    input_file: Path,
    output_file: Path,
    *,
    input_root: Path,
    output_root: Path,
    generator: LLMGenerator,
    config: GenerationConfig,
    start_line: int,
    remaining_limit: int | None,
    progress: bool,
    error_handle: Any,
) -> tuple[FileStats, int | None]:
    source_file = str(input_file.relative_to(input_root) if input_root.is_dir() else input_file.name).replace("\\", "/")
    stats = FileStats(input_file=source_file, output_file=str(output_file))
    output_file.parent.mkdir(parents=True, exist_ok=True)

    progress_bar = None
    try:
        if progress and tqdm is not None:
            progress_bar = tqdm(
                total=count_lines(input_file),
                desc=source_file,
                unit="line",
                dynamic_ncols=True,
                file=sys.stdout,
                leave=False,
            )

        with input_file.open("r", encoding="utf-8") as reader, output_file.open(
            "w", encoding="utf-8", newline="\n"
        ) as writer:
            for line_number, line in enumerate(reader, start=1):
                if progress_bar is not None:
                    progress_bar.update(1)
                if line_number < start_line or not line.strip():
                    continue
                if remaining_limit is not None and remaining_limit <= 0:
                    break

                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as exc:
                    stats.skipped += 3
                    write_error(
                        error_handle,
                        source_file=source_file,
                        source_line=line_number,
                        task_type="all",
                        generation_error=None,
                        validation_error=f"Invalid source JSON: {exc}",
                        mutation=None,
                    )
                    continue
                if not isinstance(sample, dict):
                    stats.skipped += 3
                    continue

                tool_calls = extract_tool_calls(sample)
                anchor_call = select_anchor_call(tool_calls)
                if anchor_call is None:
                    stats.skipped += 3
                    write_error(
                        error_handle,
                        source_file=source_file,
                        source_line=line_number,
                        task_type="all",
                        generation_error=None,
                        validation_error="No tool calls found in source sample.",
                        mutation=None,
                    )
                    continue

                schemas = normalize_tools(sample.get("tools", []))
                source_id = make_source_id(source_file, line_number)
                actions = build_actions(tool_calls)
                source_payload = compact_source_payload(sample, config.max_source_chars)
                hallucination_mutation = select_mutation(
                    source_id, config.seed, sample, anchor_call, schemas.get(anchor_call["name"])
                )
                disambiguation_target = select_disambiguation_target(source_id, config.seed, tool_calls, schemas)

                stats.input += 1
                if remaining_limit is not None:
                    remaining_limit -= 1

                for task_type in TASK_TYPES:
                    task_id = make_task_id(source_file, line_number, task_type)
                    mutation = hallucination_mutation if task_type == "hallucination_missing_tool" else None
                    disamb = disambiguation_target if task_type == "disambiguation_user" else None
                    record, generation_error, validation_error = generate_task_record(
                        generator,
                        task_type=task_type,
                        task_id=task_id,
                        source_id=source_id,
                        source_file=source_file,
                        source_line=line_number,
                        source_payload=source_payload,
                        original_sample=sample,
                        actions=actions,
                        anchor_call=anchor_call,
                        mutation=mutation,
                        disambiguation_target=disamb,
                        config=config,
                    )

                    if record is None:
                        if generation_error:
                            stats.failed_generation += 1
                        if validation_error:
                            stats.failed_validation += 1
                        stats.skipped += 1
                        write_error(
                            error_handle,
                            source_file=source_file,
                            source_line=line_number,
                            task_type=task_type,
                            generation_error=generation_error,
                            validation_error=validation_error,
                            mutation=mutation,
                        )
                        continue

                    writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                    writer.flush()
                    if task_type == "base":
                        stats.base_written += 1
                    elif task_type == "hallucination_missing_tool":
                        stats.hallucination_written += 1
                    elif task_type == "disambiguation_user":
                        stats.disambiguation_written += 1
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return stats, remaining_limit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Augmented ToolMind task variants from ToolMind JSONL files."
    )
    parser.add_argument("--input", default="toolmind_tool_calls_only", help="Input .jsonl file or directory.")
    parser.add_argument("--output", default="augmented_toolmind", help="Output directory.")
    parser.add_argument("--provider", choices=("openai", "nim"), required=True, help="LLM provider.")
    parser.add_argument("--model", required=True, help="Model name.")
    parser.add_argument("--api-key-env", required=True, help="Environment variable containing the API key.")
    parser.add_argument("--env-file", default=".env", help="Optional .env file to load before reading --api-key-env.")
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "OpenAI-compatible base URL. For provider=nim, defaults to "
            f"{NVIDIA_NIM_DEFAULT_BASE_URL} unless NVIDIA_NIM_BASE_URL is set."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of source samples to process.")
    parser.add_argument("--start-line", type=int, default=1, help="Start processing at this source line number.")
    parser.add_argument("--seed", type=int, default=13, help="Seed for deterministic mutation selection.")
    parser.add_argument("--allow-existing-output", action="store_true", help="Allow writing into an existing output dir.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM sampling temperature.")
    parser.add_argument("--max-output-tokens", type=int, default=4096, help="Maximum output tokens per LLM call.")
    parser.add_argument("--retry-count", type=int, default=3, help="Retries per generated task.")
    parser.add_argument(
        "--max-source-chars",
        type=int,
        default=60000,
        help="Maximum source payload characters sent to the LLM prompt.",
    )
    parser.add_argument(
        "--disable-response-format",
        action="store_true",
        help="Do not send response_format=json_object to OpenAI-compatible APIs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    env_file = Path(args.env_file).resolve()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided.")
    if args.start_line <= 0:
        raise ValueError("--start-line must be >= 1.")

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if output_root.exists() and not args.allow_existing_output:
        raise FileExistsError(
            f"Output directory already exists: {output_root}\n"
            "Use --allow-existing-output to write into it, or choose a new --output path."
        )

    files = list(iter_jsonl_files(input_path))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found under: {input_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    load_env_file(env_file)
    base_url = args.base_url
    if args.provider == "nim" and not base_url:
        base_url = os.environ.get("NVIDIA_NIM_BASE_URL") or NVIDIA_NIM_DEFAULT_BASE_URL

    config = GenerationConfig(
        provider=args.provider,
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=base_url,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        retry_count=args.retry_count,
        seed=args.seed,
        max_source_chars=args.max_source_chars,
        disable_response_format=args.disable_response_format,
    )
    generator = LLMGenerator(config)

    use_progress = not args.no_progress
    if use_progress and tqdm is None:
        print("tqdm is not installed; running without progress bars.")
        use_progress = False

    stats: list[FileStats] = []
    remaining_limit = args.limit
    error_path = output_root / "augmentation_errors.jsonl"
    file_iter: Iterable[Path] = files
    if use_progress and tqdm is not None:
        file_iter = tqdm(files, desc="Files", unit="file", dynamic_ncols=True, file=sys.stdout)

    with error_path.open("w", encoding="utf-8", newline="\n") as error_handle:
        for input_file in file_iter:
            if remaining_limit is not None and remaining_limit <= 0:
                break
            output_file = make_output_path(input_file, input_path, output_root)
            file_stats, remaining_limit = process_file(
                input_file,
                output_file,
                input_root=input_path,
                output_root=output_root,
                generator=generator,
                config=config,
                start_line=args.start_line,
                remaining_limit=remaining_limit,
                progress=use_progress,
                error_handle=error_handle,
            )
            stats.append(file_stats)

    write_stats(output_root, stats)
    print_summary(stats)
    print(f"Stats written to: {output_root / 'augmentation_stats.json'}")
    print(f"Stats written to: {output_root / 'augmentation_stats.csv'}")
    print(f"Errors written to: {error_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
