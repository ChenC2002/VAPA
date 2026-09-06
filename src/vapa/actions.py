"""Safe parser and formatter for the paper's one-action-per-turn grammar."""

from __future__ import annotations

import ast
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vapa.schemas import ActionKind

_NAMES = {
    "retrieve": ActionKind.RETRIEVE,
    "queryfield": ActionKind.QUERY_FIELD,
    "calculate": ActionKind.CALCULATE,
    "updatememory": ActionKind.UPDATE_MEMORY,
    "markstatus": ActionKind.MARK_STATUS,
    "compress": ActionKind.COMPRESS,
    "discard": ActionKind.DISCARD,
    "answer": ActionKind.ANSWER,
}

_DISPLAY = {
    ActionKind.RETRIEVE: "Retrieve",
    ActionKind.QUERY_FIELD: "QueryField",
    ActionKind.CALCULATE: "Calculate",
    ActionKind.UPDATE_MEMORY: "UpdateMemory",
    ActionKind.MARK_STATUS: "MarkStatus",
    ActionKind.COMPRESS: "Compress",
    ActionKind.DISCARD: "Discard",
    ActionKind.ANSWER: "Answer",
}

_ARGUMENTS = {
    ActionKind.RETRIEVE: ("query", "window", "domain"),
    ActionKind.QUERY_FIELD: ("field", "window"),
    ActionKind.CALCULATE: ("calculator", "bindings"),
    ActionKind.UPDATE_MEMORY: ("item",),
    ActionKind.MARK_STATUS: ("item_id", "status", "scope", "why"),
    ActionKind.COMPRESS: ("item_ids",),
    ActionKind.DISCARD: ("item_id",),
    ActionKind.ANSWER: ("prediction", "evidence"),
}


class ActionParseError(ValueError):
    pass


def action_call_prefix(kind: ActionKind) -> str:
    """Return the canonical generated-text prefix for one legal action call.

    The language-model backend uses these exact bytes to build its tokenizer-level
    feasible-action trie. ``MALFORMED`` is an environment/parser outcome rather than
    part of the policy grammar and therefore has no generatable prefix.
    """

    if not isinstance(kind, ActionKind):
        raise TypeError("action kind must be an ActionKind")
    try:
        display = _DISPLAY[kind]
    except KeyError as error:
        raise ValueError(f"{kind.value} is not part of the generated action grammar") from error
    return f"{display}("


def _validate_json_value(value: Any, path: str = "argument") -> None:
    if value is None or isinstance(value, bool | int | str):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string object key")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise TypeError(f"{path} contains non-JSON value {type(value).__name__}")


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        expected = _ARGUMENTS[self.kind]
        missing = set(expected) - set(self.arguments)
        extra = set(self.arguments) - set(expected)
        if missing or extra:
            raise ValueError(
                f"{self.kind.value} arguments mismatch; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        for name, value in self.arguments.items():
            _validate_json_value(value, name)

    def canonical(self) -> str:
        return format_action(self)


def _split_arguments(text: str) -> list[str]:
    if not text.strip():
        return []
    result: list[str] = []
    start = 0
    stack: list[str] = []
    quote: str | None = None
    escaped = False
    pairs = {"[": "]", "{": "}", "(": ")"}
    for index, character in enumerate(text):
        if escaped:
            escaped = False
            continue
        if quote is not None:
            if character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
        elif character in pairs:
            stack.append(pairs[character])
        elif character in {"}", "]", ")"}:
            if not stack or stack.pop() != character:
                raise ActionParseError("unbalanced action arguments")
        elif character == "," and not stack:
            result.append(text[start:index].strip())
            start = index + 1
    if quote is not None or stack:
        raise ActionParseError("unterminated action argument")
    result.append(text[start:].strip())
    return result


def _reject_json_constant(token: str) -> None:
    raise ActionParseError(f"non-finite JSON number is not allowed: {token}")


def _parse_json_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        _reject_json_constant(token)
    return value


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ActionParseError(f"duplicate JSON object key: {key!r}")
        output[key] = value
    return output


def _validated_literal(value: Any, source: str) -> Any:
    try:
        _validate_json_value(value)
    except (TypeError, ValueError) as exc:
        raise ActionParseError(f"invalid structured argument: {source}") from exc
    return value


def _parse_structured_value(text: str) -> Any:
    try:
        return _validated_literal(
            json.loads(
                text,
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_json_constant,
                parse_float=_parse_json_float,
            ),
            text,
        )
    except ActionParseError:
        raise
    except json.JSONDecodeError:
        pass

    # Retain the convenient single-quoted/Python-literal surface without rewriting
    # keyword-looking text inside strings. The result still has to be strict JSON data.
    try:
        return _validated_literal(ast.literal_eval(text), text)
    except (SyntaxError, ValueError) as exc:
        # The paper's evidence examples use unquoted stable pointers such as
        # ``[e#4471,e#5210]``. Parse that list recursively after both strict parsers fail.
        if text.startswith("[") and text.endswith("]"):
            inner = text[1:-1].strip()
            if not inner:
                return []
            return _validated_literal(
                [_parse_value(item) for item in _split_arguments(inner)], text
            )
        raise ActionParseError(f"invalid structured argument: {text}") from exc


_JSON_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _parse_value(text: str) -> Any:
    if text == "":
        raise ActionParseError("empty action argument")
    if text[0] in "[{(":
        return _parse_structured_value(text)
    if text[0] in "\"'":
        try:
            value = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ActionParseError(f"invalid quoted argument: {text}") from exc
        if not isinstance(value, str):
            raise ActionParseError("quoted action arguments must decode to strings")
        return value
    keywords = {"true": True, "false": False, "null": None, "none": None}
    if text.casefold() in keywords:
        return keywords[text.casefold()]
    if _JSON_NUMBER.fullmatch(text):
        if any(character in text for character in ".eE"):
            return _parse_json_float(text)
        return int(text)
    return text


def parse_action(output: str) -> Action:
    """Parse only the final non-empty line, enforcing the model surface contract."""

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise ActionParseError("empty model output")
    action_prefix = re.compile(
        r"(?i)^(?:retrieve|queryfield|calculate|updatememory|markstatus|compress|discard|answer)\s*\("
    )
    if any(action_prefix.match(line) for line in lines[:-1]):
        raise ActionParseError("an action may appear only once, on the final line")
    match = re.fullmatch(r"([A-Za-z]+)\s*\((.*)\)", lines[-1])
    if not match:
        raise ActionParseError("final line is not a valid action call")
    kind = _NAMES.get(match.group(1).lower())
    if kind is None:
        raise ActionParseError(f"unknown action: {match.group(1)}")
    raw_arguments = _split_arguments(match.group(2))
    names = _ARGUMENTS[kind]
    if len(raw_arguments) != len(names):
        raise ActionParseError(f"{_DISPLAY[kind]} expects {len(names)} arguments")
    values = {name: _parse_value(raw) for name, raw in zip(names, raw_arguments, strict=True)}
    return Action(kind, values)


def _format_value(value: Any) -> str:
    _validate_json_value(value)
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_#.%+:/ -]+", value):
        # Bare reserved words and numeric-looking strings would change type when
        # parsed. Quote those values so format_action/parse_action are inverses.
        if value.casefold() not in {"true", "false", "null", "none"} and not _JSON_NUMBER.fullmatch(
            value
        ):
            return value
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def format_action(action: Action) -> str:
    values = [_format_value(action.arguments[name]) for name in _ARGUMENTS[action.kind]]
    return f"{_DISPLAY[action.kind]}({', '.join(values)})"


def make_action(kind: ActionKind, **arguments: Any) -> Action:
    return Action(kind, arguments)
