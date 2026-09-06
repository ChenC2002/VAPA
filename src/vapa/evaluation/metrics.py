"""Dependency-free VAPA outcome and ranking metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from numbers import Real
from typing import Any, TypeAlias


class MetricInputError(ValueError):
    """Raised when a metric would otherwise be undefined or misleading."""


EvidencePointer: TypeAlias = str
GroundingPredicate: TypeAlias = Callable[
    [Sequence[EvidencePointer], Sequence[EvidencePointer]], bool
]


def _is_finite_number(value: object) -> bool:
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _finite_nonnegative(value: object, name: str) -> float:
    if not _is_finite_number(value):
        raise MetricInputError(f"{name} must be a finite number")
    result = float(value)
    if result < 0.0:
        raise MetricInputError(f"{name} cannot be negative")
    return result


def _equal_length(*values: Sequence[Any]) -> int:
    lengths = {len(value) for value in values}
    if len(lengths) != 1:
        raise MetricInputError(f"metric inputs have different lengths: {sorted(lengths)}")
    return next(iter(lengths), 0)


def exact_match(
    prediction: Any,
    reference: Any,
    *,
    normalizer: Callable[[Any], Any] | None = None,
) -> float:
    """Return binary canonical agreement.

    No text normalization is guessed.  Callers can supply the same canonicalizer
    used by their task manifest (for example, code casing or unit normalization).
    """

    if normalizer is not None:
        prediction = normalizer(prediction)
        reference = normalizer(reference)
    return float(prediction == reference)


def _evidence_pointers(values: Iterable[str], name: str) -> tuple[str, ...]:
    output = tuple(values)
    for index, value in enumerate(output):
        if not isinstance(value, str) or not value.strip():
            raise MetricInputError(f"{name}[{index}] must be a non-empty string")
    return output


def any_reference_evidence(
    predicted: Sequence[EvidencePointer],
    reference: Sequence[EvidencePointer],
) -> bool:
    """Ground an answer when it cites at least one reference pointer."""

    return bool(set(predicted) & set(reference))


def contains_all_reference_evidence(
    predicted: Sequence[EvidencePointer],
    reference: Sequence[EvidencePointer],
) -> bool:
    """Ground an answer when every required reference pointer is cited."""

    return bool(reference) and set(reference).issubset(predicted)


def exact_reference_evidence(
    predicted: Sequence[EvidencePointer],
    reference: Sequence[EvidencePointer],
) -> bool:
    """Ground an answer only when the non-empty pointer sets agree exactly."""

    return bool(reference) and set(predicted) == set(reference)


def grounded_exact_match(
    prediction: Any,
    reference: Any,
    grounded: bool | None = None,
    *,
    predicted_evidence: Iterable[EvidencePointer] = (),
    reference_evidence: Iterable[EvidencePointer] = (),
    grounding_predicate: GroundingPredicate | None = None,
    normalizer: Callable[[Any], Any] | None = None,
) -> float:
    """Return ``exact agreement × cutoff-valid grounding``.

    The paper intentionally leaves dataset-specific grounding to the manifest
    and verifier (including special handling for ``NOT_RECORDED``).  Therefore
    callers must either provide the already verified ``grounded`` flag or an
    explicit predicate over predicted/reference evidence pointers.
    """

    if grounded is not None and grounding_predicate is not None:
        raise MetricInputError("provide either grounded or grounding_predicate, not both")
    if grounded is None:
        if grounding_predicate is None:
            raise MetricInputError("grounded flag or grounding_predicate is required")
        predicted = _evidence_pointers(predicted_evidence, "predicted_evidence")
        expected = _evidence_pointers(reference_evidence, "reference_evidence")
        grounded = grounding_predicate(predicted, expected)
    if not isinstance(grounded, bool):
        raise MetricInputError("grounding result must be boolean")
    return exact_match(prediction, reference, normalizer=normalizer) * float(grounded)


def verified_match(
    prediction: object,
    reference: object,
    *,
    abs_tolerance: float = 0.0,
    relative_tolerance: float = 0.05,
) -> bool:
    """Verify a numeric answer using the paper's maximum tolerance rule.

    A prediction passes when ``|prediction-reference|`` is no greater than
    ``max(abs_tolerance, relative_tolerance * |reference|)``.
    """

    absolute = _finite_nonnegative(abs_tolerance, "abs_tolerance")
    relative = _finite_nonnegative(relative_tolerance, "relative_tolerance")
    if not _is_finite_number(prediction) or not _is_finite_number(reference):
        return False
    predicted_value = float(prediction)
    reference_value = float(reference)
    tolerance = max(absolute, relative * abs(reference_value))
    return abs(predicted_value - reference_value) <= tolerance


def _broadcast_tolerances(value: float | Iterable[float], count: int, name: str) -> list[float]:
    if _is_finite_number(value):
        return [_finite_nonnegative(value, name)] * count
    if isinstance(value, str | bytes):
        raise MetricInputError(f"{name} must be numeric")
    try:
        output = list(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise MetricInputError(f"{name} must be numeric or an iterable") from error
    if len(output) != count:
        raise MetricInputError(f"{name} has length {len(output)}; expected {count}")
    return [_finite_nonnegative(item, f"{name}[{index}]") for index, item in enumerate(output)]


def verified_accuracy(
    predictions: Iterable[object],
    references: Iterable[object],
    *,
    abs_tolerance: float | Iterable[float] = 0.0,
    relative_tolerance: float | Iterable[float] = 0.05,
    scale: float = 100.0,
) -> float:
    """Return calculation accuracy, counting parse/non-finite failures as zero."""

    predicted = list(predictions)
    expected = list(references)
    count = _equal_length(predicted, expected)
    if count == 0:
        raise MetricInputError("verified_accuracy requires at least one example")
    absolute = _broadcast_tolerances(abs_tolerance, count, "abs_tolerance")
    relative = _broadcast_tolerances(relative_tolerance, count, "relative_tolerance")
    matches = [
        verified_match(
            prediction,
            reference,
            abs_tolerance=absolute[index],
            relative_tolerance=relative[index],
        )
        for index, (prediction, reference) in enumerate(zip(predicted, expected, strict=True))
    ]
    return _scaled_mean(matches, scale=scale, name="verified_accuracy")


def _success_values(successes: Iterable[float | bool]) -> list[float]:
    output: list[float] = []
    for index, success in enumerate(successes):
        if isinstance(success, bool):
            value = float(success)
        elif _is_finite_number(success):
            value = float(success)
        else:
            raise MetricInputError(f"successes[{index}] must be finite")
        if not 0.0 <= value <= 1.0:
            raise MetricInputError(f"successes[{index}] must be in [0, 1]")
        output.append(value)
    return output


def _scaled_mean(values: Iterable[float | bool], *, scale: float, name: str) -> float:
    scale_value = _finite_nonnegative(scale, "scale")
    values_list = [float(value) for value in values]
    if not values_list:
        raise MetricInputError(f"{name} requires at least one example")
    return scale_value * math.fsum(values_list) / len(values_list)


def task_success(successes: Iterable[float | bool], *, scale: float = 100.0) -> float:
    """Task success (TS): ``scale / N * sum(success_i)``."""

    return _scaled_mean(_success_values(successes), scale=scale, name="task_success")


def cost_normalized_contributions(
    successes: Iterable[float | bool],
    costs: Iterable[float],
    max_costs: Iterable[float],
) -> list[float]:
    """Return per-example ``success * (1 - cost / max_cost)`` values."""

    success_values = _success_values(successes)
    cost_values = list(costs)
    maximum_values = list(max_costs)
    _equal_length(success_values, cost_values, maximum_values)
    output: list[float] = []
    for index, (success, cost, maximum) in enumerate(
        zip(success_values, cost_values, maximum_values, strict=True)
    ):
        cost_value = _finite_nonnegative(cost, f"costs[{index}]")
        maximum_value = _finite_nonnegative(maximum, f"max_costs[{index}]")
        if maximum_value <= 0.0:
            raise MetricInputError(f"max_costs[{index}] must be positive")
        if cost_value > maximum_value:
            raise MetricInputError(f"costs[{index}] exceeds max_costs[{index}]")
        output.append(success * (1.0 - cost_value / maximum_value))
    return output


def cost_normalized_success(
    successes: Iterable[float | bool],
    costs: Iterable[float],
    max_costs: Iterable[float],
    *,
    scale: float = 100.0,
) -> float:
    """Cost-normalized success (CNS), reported as a percentage by default."""

    values = cost_normalized_contributions(successes, costs, max_costs)
    return _scaled_mean(values, scale=scale, name="cost_normalized_success")


def macro_task_average(
    values: Iterable[float],
    task_ids: Iterable[Hashable],
    *,
    task_weights: Mapping[Hashable, float] | None = None,
) -> float:
    """Average examples within tasks, then give each task equal weight by default."""

    value_list = list(values)
    task_list = list(task_ids)
    count = _equal_length(value_list, task_list)
    if count == 0:
        raise MetricInputError("macro_task_average requires at least one example")

    grouped: dict[Hashable, list[float]] = defaultdict(list)
    for index, (value, task_id) in enumerate(zip(value_list, task_list, strict=True)):
        if not isinstance(task_id, Hashable):
            raise MetricInputError(f"task_ids[{index}] is not hashable")
        if not _is_finite_number(value) and not isinstance(value, bool):
            raise MetricInputError(f"values[{index}] must be finite")
        grouped[task_id].append(float(value))
    task_means = {
        task_id: math.fsum(task_values) / len(task_values)
        for task_id, task_values in grouped.items()
    }

    if task_weights is None:
        return math.fsum(task_means.values()) / len(task_means)
    unknown = set(task_weights) - set(task_means)
    missing = set(task_means) - set(task_weights)
    if unknown or missing:
        raise MetricInputError(f"task weight keys mismatch: missing={missing}, unknown={unknown}")
    weights = {
        task_id: _finite_nonnegative(task_weights[task_id], f"task_weights[{task_id!r}]")
        for task_id in task_means
    }
    total_weight = math.fsum(weights.values())
    if total_weight <= 0.0:
        raise MetricInputError("task weights must have positive total weight")
    weighted_sum = math.fsum(task_means[task_id] * weights[task_id] for task_id in task_means)
    return weighted_sum / total_weight


def macro_task_success(
    successes: Iterable[float | bool],
    task_ids: Iterable[Hashable],
    *,
    scale: float = 100.0,
) -> float:
    """Macro TS with every task type weighted equally."""

    return _finite_nonnegative(scale, "scale") * macro_task_average(
        _success_values(successes), task_ids
    )


def macro_cost_normalized_success(
    successes: Iterable[float | bool],
    costs: Iterable[float],
    max_costs: Iterable[float],
    task_ids: Iterable[Hashable],
    *,
    scale: float = 100.0,
) -> float:
    """Macro CNS with every task type weighted equally."""

    contributions = cost_normalized_contributions(successes, costs, max_costs)
    return _finite_nonnegative(scale, "scale") * macro_task_average(contributions, task_ids)


def _binary_inputs(
    labels: Iterable[int | bool], scores: Iterable[float]
) -> list[tuple[int, float]]:
    label_list = list(labels)
    score_list = list(scores)
    count = _equal_length(label_list, score_list)
    if count == 0:
        raise MetricInputError("ranking metric requires at least one example")
    pairs: list[tuple[int, float]] = []
    for index, (label, score) in enumerate(zip(label_list, score_list, strict=True)):
        if label not in (0, 1, False, True):
            raise MetricInputError(f"labels[{index}] must be binary")
        if not _is_finite_number(score):
            raise MetricInputError(f"scores[{index}] must be finite")
        pairs.append((int(label), float(score)))
    return pairs


def auroc(labels: Iterable[int | bool], scores: Iterable[float]) -> float:
    """Area under the ROC curve using average ranks for tied scores."""

    pairs = sorted(_binary_inputs(labels, scores), key=lambda pair: pair[1])
    positives = sum(label for label, _ in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        raise MetricInputError("AUROC requires both positive and negative examples")

    positive_rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][1] == pairs[index][1]:
            end += 1
        # Ranks are one-based; every item in a tie receives the average rank.
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(label for label, _ in pairs[index:end])
        index = end
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def auprc(labels: Iterable[int | bool], scores: Iterable[float]) -> float:
    """Step-integrated precision-recall area with score ties updated as a group.

    This is the standard non-interpolated average-precision definition.  Grouping
    equal scores makes the result independent of input ordering within ties.
    """

    pairs = sorted(_binary_inputs(labels, scores), key=lambda pair: pair[1], reverse=True)
    positives = sum(label for label, _ in pairs)
    if positives == 0:
        raise MetricInputError("AUPRC requires at least one positive example")

    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    area = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][1] == pairs[index][1]:
            end += 1
        group_positives = sum(label for label, _ in pairs[index:end])
        true_positives += group_positives
        false_positives += (end - index) - group_positives
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        index = end
    return area


# Common terminology used by evaluation libraries and earlier project scripts.
average_precision = auprc
