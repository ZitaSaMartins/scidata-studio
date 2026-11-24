"""Statistics for sensory discrimination tests (no Streamlit).

Covers the scoring of response sheets and the Thurstonian estimates reported
by XLSTAT's "Sensory discrimination tests": proportion of correct answers,
proportion of discriminators and d-prime, each with a Clopper-Pearson based
confidence interval, plus the exact binomial p-value and power.

``tests/test_discrimination_stats.py`` checks these numbers against the XLSTAT
output stored in ``Demo_TRI_EN.xlsm``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import integrate, optimize, stats

_D_MAX = 15.0


def _triangle(d: float) -> float:
    def integrand(z: float) -> float:
        return (
            stats.norm.cdf(-z * np.sqrt(3) + d * np.sqrt(2 / 3))
            + stats.norm.cdf(-z * np.sqrt(3) - d * np.sqrt(2 / 3))
        ) * stats.norm.pdf(z)

    return float(2 * integrate.quad(integrand, 0, np.inf)[0])


def _duo_trio(d: float) -> float:
    a = stats.norm.cdf(d / np.sqrt(2))
    b = stats.norm.cdf(d / np.sqrt(6))
    return float(1 - a - b + 2 * a * b)


def _two_afc(d: float) -> float:
    return float(stats.norm.cdf(d / np.sqrt(2)))


def _tetrad(d: float) -> float:
    def integrand(z: float) -> float:
        shifted = stats.norm.cdf(z - d)
        return stats.norm.pdf(z) * (2 * stats.norm.cdf(z) * shifted - shifted**2)

    return float(1 - 2 * integrate.quad(integrand, -np.inf, np.inf)[0])


@dataclass(frozen=True)
class DiscriminationMethod:
    name: str
    guess: float
    psychometric: Callable[[float], float]


METHODS: dict[str, DiscriminationMethod] = {
    "triangle": DiscriminationMethod("Triangle test", 1 / 3, _triangle),
    "duotrio": DiscriminationMethod("Duo-trio test", 1 / 2, _duo_trio),
    "paired": DiscriminationMethod("Paired comparison (2-AFC)", 1 / 2, _two_afc),
    "tetrad": DiscriminationMethod("Tetrad test", 1 / 3, _tetrad),
}


def proportion_correct(method: DiscriminationMethod, d_prime_value: float) -> float:
    """Expected proportion of correct answers for a given d-prime."""
    return method.psychometric(d_prime_value)


def d_prime(method: DiscriminationMethod, pc: float) -> float:
    """Invert the psychometric function. Returns 0 at or below chance, inf at 1."""
    if pc <= method.guess:
        return 0.0
    if pc >= method.psychometric(_D_MAX):
        return float("inf")
    return float(optimize.brentq(lambda d: method.psychometric(d) - pc, 0.0, _D_MAX))


def _normalize(values: pd.Series) -> pd.Series:
    def clean(value: object) -> object:
        if pd.isna(value):
            return np.nan
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return str(value).strip().casefold()

    return values.map(clean)


def score_responses(answers: pd.Series, correct: pd.Series) -> pd.Series:
    """Per-row correctness: True/False, or NA when either value is missing.

    Codes are compared as trimmed, case-insensitive text so that ``123``,
    ``123.0`` and ``" 123"`` all match.
    """
    given = _normalize(answers)
    expected = _normalize(correct)
    result = (given == expected).astype("boolean")
    result[given.isna() | expected.isna()] = pd.NA
    return result


def answers_outside_samples(
    df: pd.DataFrame, sample_columns: list[str], value_column: str
) -> pd.Series:
    """Rows whose ``value_column`` is not one of the samples served in that row."""
    value = _normalize(df[value_column])
    served = pd.concat([_normalize(df[column]) for column in sample_columns], axis=1)
    inside = served.eq(value, axis=0).any(axis=1)
    return value.notna() & ~inside


def critical_correct(total: int, guess: float, alpha: float) -> int | None:
    """Smallest number of correct answers that is significant, or None."""
    for count in range(total + 1):
        if stats.binomtest(count, total, guess, alternative="greater").pvalue < alpha:
            return count
    return None


def discrimination_summary(
    successes: int, total: int, method: DiscriminationMethod, alpha: float = 0.05
) -> tuple[dict[str, float | int | None], pd.DataFrame]:
    """Test results and estimated parameters, laid out like XLSTAT.

    Returns ``(test, parameters)``: ``test`` holds the statistic, p-value,
    critical count and power; ``parameters`` has one row each for the
    proportion of correct answers, the proportion of discriminators and
    d-prime, with standard deviation and a two-sided Clopper-Pearson interval.
    """
    guess = method.guess
    pc = successes / total
    result = stats.binomtest(successes, total, guess, alternative="greater")
    critical = critical_correct(total, guess, alpha)
    power = float(stats.binom.sf(critical - 1, total, pc)) if critical is not None else 0.0

    ci = stats.binomtest(successes, total, guess).proportion_ci(confidence_level=1 - alpha)
    pc_low, pc_high = float(ci.low), float(ci.high)
    pc_sd = float(np.sqrt(pc * (1 - pc) / total))

    def to_pd(value: float) -> float:
        return max(0.0, (value - guess) / (1 - guess))

    d_value = d_prime(method, pc)
    if 0 < d_value < np.inf:
        step = 1e-5
        slope = (method.psychometric(d_value + step) - method.psychometric(d_value - step)) / (2 * step)
        d_sd = pc_sd / slope
    else:
        d_sd = float("nan")

    parameters = pd.DataFrame(
        {
            "Value": [pc, to_pd(pc), d_value],
            "Std. deviation": [pc_sd, pc_sd / (1 - guess), d_sd],
            "Lower bound": [pc_low, to_pd(pc_low), d_prime(method, pc_low)],
            "Upper bound": [pc_high, to_pd(pc_high), d_prime(method, pc_high)],
        },
        index=pd.Index(["Prop. of correct answers", "Prop. of discriminators", "d-prime"], name="Parameter"),
    )
    test = {
        "Correct": successes,
        "Total": total,
        "Guessing probability": guess,
        "d-prime": d_value,
        "p-value": float(result.pvalue),
        "Critical correct": critical,
        "Power": power,
    }
    return test, parameters
