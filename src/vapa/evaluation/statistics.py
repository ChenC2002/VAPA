"""Small deterministic helpers for the paper's paired-seed analysis."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float
    n: int


@dataclass(frozen=True)
class PairedTest:
    estimate: float
    lower: float
    upper: float
    n: int
    t_statistic: float
    p_value: float


def _t_critical(df: int) -> float:
    # Invert the same two-sided Student-t distribution used for p values.
    def tail(t: float) -> float:
        return _regularized_incomplete_beta(df / 2.0, 0.5, df / (df + t * t))

    low, high = 0.0, 1.0
    while tail(high) > 0.05:
        high *= 2.0
    for _ in range(70):
        middle = (low + high) / 2.0
        if tail(middle) > 0.05:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def paired_t_interval(differences: Iterable[float]) -> Interval:
    values = [float(value) for value in differences]
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        raise ValueError("paired interval requires at least two finite seed differences")
    estimate = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    half_width = _t_critical(len(values) - 1) * standard_error
    return Interval(estimate, estimate - half_width, estimate + half_width, len(values))


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    maximum_iterations = 200
    epsilon = 3e-14
    minimum = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < minimum:
        d = minimum
    d = 1.0 / d
    result = d
    for iteration in range(1, maximum_iterations + 1):
        even = 2 * iteration
        coefficient = iteration * (b - iteration) * x / ((qam + even) * (a + even))
        d = 1.0 + coefficient * d
        if abs(d) < minimum:
            d = minimum
        c = 1.0 + coefficient / c
        if abs(c) < minimum:
            c = minimum
        d = 1.0 / d
        result *= d * c
        coefficient = -(a + iteration) * (qab + iteration) * x / ((a + even) * (qap + even))
        d = 1.0 + coefficient * d
        if abs(d) < minimum:
            d = minimum
        c = 1.0 + coefficient / c
        if abs(c) < minimum:
            c = minimum
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= epsilon:
            return result
    raise ArithmeticError("incomplete-beta continued fraction did not converge")


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if not 0.0 <= x <= 1.0 or a <= 0.0 or b <= 0.0:
        raise ValueError("invalid incomplete-beta arguments")
    if x in {0.0, 1.0}:
        return x
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def paired_t_test(differences: Iterable[float]) -> PairedTest:
    """Return the paired mean contrast, 95% interval, and two-sided t-test p value."""

    values = [float(value) for value in differences]
    interval = paired_t_interval(values)
    standard_deviation = statistics.stdev(values)
    if standard_deviation == 0.0:
        statistic = 0.0 if interval.estimate == 0.0 else math.copysign(math.inf, interval.estimate)
        p_value = 1.0 if interval.estimate == 0.0 else 0.0
    else:
        statistic = interval.estimate / (standard_deviation / math.sqrt(interval.n))
        degrees_of_freedom = interval.n - 1
        x = degrees_of_freedom / (degrees_of_freedom + statistic * statistic)
        p_value = _regularized_incomplete_beta(degrees_of_freedom / 2.0, 0.5, x)
    return PairedTest(
        estimate=interval.estimate,
        lower=interval.lower,
        upper=interval.upper,
        n=interval.n,
        t_statistic=statistic,
        p_value=max(0.0, min(1.0, p_value)),
    )


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return monotone Holm-adjusted p values for one prespecified family."""

    if not p_values:
        return {}
    for name, value in p_values.items():
        if not name or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Holm inputs require named finite p values in [0, 1]")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[name] = running
    return adjusted


def ordinary_least_squares_slope(x: Iterable[float], y: Iterable[float]) -> float:
    x_values = [float(value) for value in x]
    y_values = [float(value) for value in y]
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("slope requires aligned inputs with at least two points")
    if not all(math.isfinite(value) for value in x_values + y_values):
        raise ValueError("slope inputs must be finite")
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator == 0:
        raise ValueError("slope is undefined for constant x")
    return (
        sum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(x_values, y_values, strict=True)
        )
        / denominator
    )
