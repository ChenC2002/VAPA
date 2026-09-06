"""Manifest-driven deterministic calculator resolution.

The engine is intentionally small and safe: formulas are parsed into a restricted AST,
never passed to Python's ``eval``.  Paper-specific formulas and operationalization rules
belong in a separately versioned manifest, because the PDF does not publish those bytes.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vapa.artifacts import strict_json_loads
from vapa.schemas import MemoryItem, RecordEvent, ReturnCode, ToolReturn, normalize_field


@dataclass(frozen=True)
class InputSpec:
    name: str
    fields: tuple[str, ...]
    unit: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", normalize_field(self.name))
        object.__setattr__(self, "fields", tuple(normalize_field(item) for item in self.fields))
        if not self.fields:
            raise ValueError("calculator inputs require at least one source field")


@dataclass(frozen=True)
class CalculatorSpec:
    name: str
    inputs: tuple[InputSpec, ...]
    expression: str
    output_unit: str | None = None
    output_decimals: int = 2
    absolute_tolerance: float = 0.01

    def __post_init__(self) -> None:
        normalized = normalize_field(self.name)
        object.__setattr__(self, "name", normalized)
        names = [item.name for item in self.inputs]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate input in calculator {normalized}")
        if not self.expression.strip():
            raise ValueError("calculator expression cannot be empty")
        if self.output_decimals < 0:
            raise ValueError("output_decimals cannot be negative")
        if self.absolute_tolerance < 0:
            raise ValueError("absolute_tolerance cannot be negative")
        _validate_expression(self.expression, set(names))


_BINARY = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
    ast.Mod: lambda a, b: a % b,
}
_UNARY = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a}
_FUNCTIONS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
}


def _validate_expression(expression: str, inputs: set[str]) -> None:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid calculator expression: {expression}") from exc
    allowed = (
        ast.Expression,
        ast.Constant,
        ast.Name,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Load,
        *tuple(_BINARY),
        *tuple(_UNARY),
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(f"unsupported calculator syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in inputs | set(_FUNCTIONS):
            raise ValueError(f"unknown calculator symbol: {node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
                raise ValueError("calculator calls an unsupported function")
            if node.keywords:
                raise ValueError("calculator functions do not accept keyword arguments")


def _evaluate_node(node: ast.AST, values: Mapping[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body, values)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.Name):
        return float(values[node.id])
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return float(
            _BINARY[type(node.op)](
                _evaluate_node(node.left, values), _evaluate_node(node.right, values)
            )
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return float(_UNARY[type(node.op)](_evaluate_node(node.operand, values)))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        arguments = [_evaluate_node(item, values) for item in node.args]
        return float(_FUNCTIONS[node.func.id](*arguments))
    raise ValueError(f"unsupported calculator node: {type(node).__name__}")


def evaluate_expression(expression: str, values: Mapping[str, float]) -> float:
    result = _evaluate_node(ast.parse(expression, mode="eval"), values)
    if not math.isfinite(result):
        raise ValueError("calculator result is not finite")
    return result


class UnitConverter:
    def __init__(self) -> None:
        self._affine: dict[tuple[str, str], tuple[float, float]] = {}

    def register(self, source: str, target: str, *, factor: float, offset: float = 0.0) -> None:
        self._affine[(source.strip().lower(), target.strip().lower())] = (factor, offset)

    def convert(self, value: float, source: str | None, target: str | None) -> float:
        if target is None:
            return value
        if source is None or not source.strip():
            raise ValueError(f"source unit is required for conversion to {target!r}")
        if source.strip().lower() == target.strip().lower():
            return value
        rule = self._affine.get((source.strip().lower(), target.strip().lower()))
        if rule is None:
            raise ValueError(f"no unit conversion from {source!r} to {target!r}")
        factor, offset = rule
        return value * factor + offset


class CalculatorRegistry:
    def __init__(self, specs: Iterable[CalculatorSpec] = ()) -> None:
        self._specs: dict[str, CalculatorSpec] = {}
        self.units = UnitConverter()
        self.units.register("lb", "kg", factor=0.45359237)
        self.units.register("kg", "lb", factor=2.2046226218)
        self.units.register("cm", "m", factor=0.01)
        self.units.register("m", "cm", factor=100.0)
        for spec in specs:
            self.register(spec)

    def register(self, spec: CalculatorSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"duplicate calculator: {spec.name}")
        self._specs[spec.name] = spec

    def __contains__(self, name: str) -> bool:
        return normalize_field(name) in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def get(self, name: str) -> CalculatorSpec:
        try:
            return self._specs[normalize_field(name)]
        except KeyError as exc:
            raise KeyError(f"unknown calculator: {name}") from exc

    @classmethod
    def from_json(cls, path: str | Path) -> CalculatorRegistry:
        payload = strict_json_loads(Path(path).read_bytes())
        return cls.from_manifest(payload)

    @classmethod
    def from_manifest(cls, payload: Any) -> CalculatorRegistry:
        """Build from an already strictly decoded calculator manifest."""

        if isinstance(payload, dict):
            allowed_top = {"schema_version", "paper_exact", "calculators"}
            extra_top = set(payload) - allowed_top
            if extra_top:
                raise ValueError(f"unknown calculator manifest keys: {sorted(extra_top)}")
            if payload.get("schema_version") != 1:
                raise ValueError("calculator manifest schema_version must be 1")
            if not isinstance(payload.get("paper_exact"), bool):
                raise ValueError("calculator manifest paper_exact flag must be boolean")
            rows = payload.get("calculators")
        else:
            rows = payload
        if not isinstance(rows, list):
            raise ValueError("calculator manifest must contain a list")
        specs: list[CalculatorSpec] = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("each calculator must be an object")
            allowed = {
                "name",
                "inputs",
                "expression",
                "output_unit",
                "output_decimals",
                "absolute_tolerance",
            }
            extra = set(row) - allowed
            if extra:
                raise ValueError(f"unknown calculator keys: {sorted(extra)}")
            missing = {"name", "inputs", "expression"} - set(row)
            if missing:
                raise ValueError(f"calculator is missing keys: {sorted(missing)}")
            if not isinstance(row["inputs"], list):
                raise ValueError("calculator inputs must be an array")
            inputs = tuple(
                InputSpec(
                    name=str(item["name"]),
                    fields=tuple(str(value) for value in item.get("fields", [item["name"]])),
                    unit=None if item.get("unit") is None else str(item["unit"]),
                )
                for item in row["inputs"]
            )
            specs.append(
                CalculatorSpec(
                    name=str(row["name"]),
                    inputs=inputs,
                    expression=str(row["expression"]),
                    output_unit=row.get("output_unit"),
                    output_decimals=int(row.get("output_decimals", 2)),
                    absolute_tolerance=float(row.get("absolute_tolerance", 0.01)),
                )
            )
        return cls(specs)

    def calculate(
        self,
        name: str,
        bindings: Mapping[str, Any],
        *,
        events: Mapping[str, RecordEvent],
        memory: Mapping[str, MemoryItem],
    ) -> ToolReturn:
        try:
            spec = self.get(name)
        except (KeyError, ValueError):
            return ToolReturn(ReturnCode.UNRESOLVED, message=f"unknown calculator {name}")
        expected = {item.name for item in spec.inputs}
        normalized_bindings = {normalize_field(str(key)): value for key, value in bindings.items()}
        if set(normalized_bindings) != expected:
            return ToolReturn(
                ReturnCode.UNBOUND, message="bindings must exactly match calculator inputs"
            )
        values: dict[str, float] = {}
        pointers: list[str] = []
        try:
            for input_spec in spec.inputs:
                source = normalized_bindings[input_spec.name]
                if isinstance(source, Mapping):
                    source = source.get("pointer", source.get("id"))
                source_id = str(source)
                if source_id in events:
                    event = events[source_id]
                    field = event.field
                    raw_value = event.value
                    unit = event.unit
                    source_pointers = (event.pointer,)
                elif source_id in memory:
                    item = memory[source_id]
                    field = item.field
                    raw_value = item.value
                    unit = item.unit
                    source_pointers = item.evidence_pointers
                else:
                    return ToolReturn(
                        ReturnCode.UNRESOLVED, message=f"binding {source_id} does not resolve"
                    )
                if field not in input_spec.fields:
                    return ToolReturn(
                        ReturnCode.UNRESOLVED,
                        message=f"{input_spec.name} cannot bind field {field}",
                    )
                if isinstance(raw_value, tuple) or not isinstance(raw_value, int | float):
                    return ToolReturn(ReturnCode.UNRESOLVED, message=f"{source_id} is not numeric")
                values[input_spec.name] = self.units.convert(
                    float(raw_value), unit, input_spec.unit
                )
                pointers.extend(source_pointers)
            raw_result = evaluate_expression(spec.expression, values)
        except (ArithmeticError, OverflowError, ValueError) as exc:
            return ToolReturn(ReturnCode.AMBIGUOUS, message=str(exc))
        rounded = round(raw_result, spec.output_decimals)
        return ToolReturn(
            ReturnCode.VALUE,
            value=rounded,
            unit=spec.output_unit,
            evidence_pointers=tuple(dict.fromkeys(pointers)),
        )
