"""Deterministic patient-disjoint dataset splitting."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from numbers import Real
from typing import Any

from vapa.data.io import DataValidationError

_HASH_SPACE = 1 << 256
_DEFAULT_NAMESPACE = "vapa-patient-split-v1"


@dataclass(frozen=True)
class SplitFractions:
    """Fractions for the train, validation, and test partitions."""

    train: float = 0.8
    validation: float = 0.1
    test: float = 0.1

    def decimals(self) -> tuple[Decimal, Decimal, Decimal]:
        raw_values = (self.train, self.validation, self.test)
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in raw_values
        ):
            raise ValueError("split fractions must be finite numbers")
        values = tuple(Decimal(str(value)) for value in raw_values)
        if any(not value.is_finite() or value < 0 for value in values):
            raise ValueError("split fractions must be finite and non-negative")
        if sum(values) != Decimal(1):
            raise ValueError("split fractions must sum exactly to 1")
        return values[0], values[1], values[2]


_DEFAULT_FRACTIONS = SplitFractions()


def _patient_id(record: Mapping[str, Any], field: str, location: str) -> str | int:
    if field not in record:
        raise DataValidationError(f"{location} is missing patient ID field {field!r}")
    value = record[field]
    if isinstance(value, bool) or not isinstance(value, str | int):
        raise DataValidationError(f"{location} field {field!r} must be a string or integer")
    if isinstance(value, str) and not value.strip():
        raise DataValidationError(f"{location} field {field!r} cannot be empty")
    return value


def _typed_id(value: str | int) -> tuple[str, str | int]:
    return type(value).__name__, value


def patient_hash(
    patient_id: str | int,
    *,
    seed: int = 0,
    namespace: str = _DEFAULT_NAMESPACE,
) -> str:
    """Return the stable SHA-256 used to assign one patient to one split."""

    if isinstance(patient_id, bool) or not isinstance(patient_id, str | int):
        raise TypeError("patient_id must be a string or integer")
    if isinstance(patient_id, str) and not patient_id.strip():
        raise ValueError("patient_id cannot be empty")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace cannot be empty")

    payload = json.dumps(
        {
            "namespace": namespace,
            "patient_id": patient_id,
            "patient_id_type": type(patient_id).__name__,
            "seed": seed,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assign_patient_split(
    patient_id: str | int,
    *,
    fractions: SplitFractions = _DEFAULT_FRACTIONS,
    seed: int = 0,
    namespace: str = _DEFAULT_NAMESPACE,
) -> str:
    """Assign a patient using exact integer boundaries in SHA-256 space."""

    train, validation, _ = fractions.decimals()
    train_end = int((train * _HASH_SPACE).to_integral_value(rounding=ROUND_FLOOR))
    validation_end = int(
        ((train + validation) * _HASH_SPACE).to_integral_value(rounding=ROUND_FLOOR)
    )
    bucket = int(patient_hash(patient_id, seed=seed, namespace=namespace), 16)
    if bucket < train_end:
        return "train"
    if bucket < validation_end:
        return "validation"
    return "test"


def validate_patient_disjoint(
    splits: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    patient_id_field: str = "patient_id",
    expected_patient_ids: Iterable[str | int] | None = None,
) -> None:
    """Reject any patient appearing in two splits.

    If ``expected_patient_ids`` is supplied, this also proves that the split
    covers exactly that patient set—neither dropped nor fabricated patients are
    accepted.
    """

    owner: dict[tuple[str, str | int], str] = {}
    actual: set[tuple[str, str | int]] = set()
    for split_name, records in splits.items():
        if not isinstance(split_name, str) or not split_name:
            raise DataValidationError("split names must be non-empty strings")
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise DataValidationError(f"{split_name}[{index}] must be a mapping")
            value = _patient_id(record, patient_id_field, f"{split_name}[{index}]")
            key = _typed_id(value)
            previous = owner.get(key)
            if previous is not None and previous != split_name:
                raise DataValidationError(
                    f"patient {value!r} leaks across splits {previous!r} and {split_name!r}"
                )
            owner[key] = split_name
            actual.add(key)

    if expected_patient_ids is not None:
        expected: set[tuple[str, str | int]] = set()
        for index, value in enumerate(expected_patient_ids):
            if isinstance(value, bool) or not isinstance(value, str | int):
                raise DataValidationError(
                    f"expected_patient_ids[{index}] must be a string or integer"
                )
            if isinstance(value, str) and not value.strip():
                raise DataValidationError(f"expected_patient_ids[{index}] cannot be empty")
            expected.add(_typed_id(value))
        missing = expected - actual
        unexpected = actual - expected
        if missing or unexpected:
            raise DataValidationError(
                "split patient coverage mismatch: "
                f"missing={sorted(map(repr, missing))}, unexpected={sorted(map(repr, unexpected))}"
            )


def patient_disjoint_hash_split(
    records: Iterable[Mapping[str, Any]],
    *,
    patient_id_field: str = "patient_id",
    fractions: SplitFractions = _DEFAULT_FRACTIONS,
    seed: int = 0,
    namespace: str = _DEFAULT_NAMESPACE,
) -> dict[str, list[Mapping[str, Any]]]:
    """Split records deterministically while keeping every patient intact.

    Input order is preserved within each output partition.  Assignment depends
    only on the typed patient ID, namespace, seed, and fractions; adding a new
    patient never reshuffles existing patients.
    """

    fractions.decimals()
    output: dict[str, list[Mapping[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    expected: set[tuple[str, str | int]] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise DataValidationError(f"record {index} must be a mapping")
        value = _patient_id(record, patient_id_field, f"record {index}")
        expected.add(_typed_id(value))
        split_name = assign_patient_split(
            value,
            fractions=fractions,
            seed=seed,
            namespace=namespace,
        )
        output[split_name].append(record)

    expected_values = [value for _, value in expected]
    validate_patient_disjoint(
        output,
        patient_id_field=patient_id_field,
        expected_patient_ids=expected_values,
    )
    return output


# Familiar concise names for callers that already make the patient constraint explicit.
hash_split = patient_disjoint_hash_split
assert_patient_disjoint = validate_patient_disjoint
