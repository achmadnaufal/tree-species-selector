"""
Species diversity scorer for proposed planting plans.

This module computes ecological diversity indices (Shannon, Simpson) and a
functional diversity score for a planting plan represented as a DataFrame of
species with their proportional abundances and functional traits.

All public functions are pure and immutable — inputs are never modified;
every function returns a new object.

Author: github.com/achmadnaufal
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum proportion value treated as non-zero (avoids log(0)).
_MIN_PROPORTION: float = 1e-10

#: Columns that are treated as functional traits for functional diversity.
DEFAULT_TRAIT_COLUMNS: tuple[str, ...] = (
    "growth_rate_m_yr",
    "carbon_seq_tc_ha_yr",
    "min_rainfall_mm",
    "max_rainfall_mm",
    "min_temp_c",
    "max_temp_c",
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiversityResult:
    """Immutable container for diversity index results.

    Attributes:
        shannon_index: Shannon entropy H' (nats).  Higher values indicate
            greater diversity.  Zero when all proportion is in one species.
        simpson_index: Simpson diversity index D (1 - sum p_i^2).  Ranges
            [0, 1); values closer to 1 indicate higher diversity.
        evenness: Pielou's evenness J' = H' / ln(S).  Ranges [0, 1].
            Returns ``None`` when fewer than two species are present.
        functional_diversity: Mean pairwise Euclidean distance between
            species in normalised trait space.  Returns ``None`` when trait
            columns are absent or when fewer than two species are present.
        species_count: Number of species with non-zero proportion.
        summary: Human-readable summary string.
    """

    shannon_index: float
    simpson_index: float
    evenness: Optional[float]
    functional_diversity: Optional[float]
    species_count: int
    summary: str = field(compare=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_plan(plan: pd.DataFrame, proportion_col: str) -> None:
    """Validate the planting-plan DataFrame at the system boundary.

    Args:
        plan: DataFrame representing the planting plan.
        proportion_col: Name of the column containing species proportions.

    Raises:
        TypeError: If ``plan`` is not a :class:`pandas.DataFrame`.
        ValueError: If ``plan`` is empty, the proportion column is missing,
            any proportion is negative, or all proportions are zero.
    """
    if not isinstance(plan, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame, got {type(plan).__name__!r}."
        )
    if plan.empty:
        raise ValueError("Planting plan DataFrame is empty.")
    if proportion_col not in plan.columns:
        raise ValueError(
            f"Proportion column {proportion_col!r} not found in DataFrame. "
            f"Available columns: {list(plan.columns)}."
        )
    proportions = plan[proportion_col]
    if (proportions < 0).any():
        raise ValueError(
            f"Column {proportion_col!r} contains negative values. "
            "All proportions must be >= 0."
        )
    if proportions.sum() == 0:
        raise ValueError(
            f"All values in {proportion_col!r} are zero; "
            "cannot compute diversity on an empty plan."
        )


def _normalise_proportions(proportions: pd.Series) -> pd.Series:
    """Return a new Series with values normalised to sum to 1.

    Rows with proportion <= _MIN_PROPORTION are included but treated as
    effectively absent for log-based computations.

    Args:
        proportions: Raw proportion or area values (non-negative).

    Returns:
        New Series where values sum to 1.0.
    """
    total = proportions.sum()
    return proportions / total


def _shannon(p: pd.Series) -> float:
    """Compute Shannon entropy H' from normalised proportions.

    Args:
        p: Series of normalised proportions summing to 1.

    Returns:
        Shannon entropy in nats (natural logarithm base).
    """
    active = p[p > _MIN_PROPORTION]
    return float(-(active * active.apply(math.log)).sum())


def _simpson(p: pd.Series) -> float:
    """Compute Simpson diversity index D = 1 - sum(p_i^2).

    Args:
        p: Series of normalised proportions summing to 1.

    Returns:
        Simpson diversity index in range [0, 1).
    """
    return float(1.0 - (p**2).sum())


def _pielou_evenness(shannon: float, species_count: int) -> Optional[float]:
    """Compute Pielou's evenness J' = H' / ln(S).

    Args:
        shannon: Shannon entropy H'.
        species_count: Number of species with non-zero proportion.

    Returns:
        Evenness value in [0, 1], or ``None`` if fewer than 2 species.
    """
    if species_count < 2:
        return None
    max_shannon = math.log(species_count)
    if max_shannon == 0:
        return None
    return float(shannon / max_shannon)


def _functional_diversity(
    plan: pd.DataFrame,
    trait_columns: tuple[str, ...],
) -> Optional[float]:
    """Compute mean pairwise Euclidean distance in normalised trait space.

    Only columns that are present in *plan* and contain numeric data are used.
    If fewer than two species are present or no valid trait columns exist,
    ``None`` is returned.

    Args:
        plan: Planting plan DataFrame (rows = species).
        trait_columns: Column names to use as functional traits.

    Returns:
        Mean pairwise Euclidean distance, or ``None`` when not computable.
    """
    available = [
        c for c in trait_columns if c in plan.columns and pd.api.types.is_numeric_dtype(plan[c])
    ]
    if not available or len(plan) < 2:
        return None

    trait_df = plan[available].copy()

    # Drop rows with all-NaN traits
    trait_df = trait_df.dropna(how="all")
    if len(trait_df) < 2:
        return None

    # Fill remaining NaN with column mean (imputation — does not mutate input)
    trait_df = trait_df.fillna(trait_df.mean())

    # Min-max normalise each trait column to [0, 1]
    col_min = trait_df.min()
    col_max = trait_df.max()
    col_range = col_max - col_min

    # Avoid division by zero for constant columns
    safe_range = col_range.replace(0, 1)
    normed = (trait_df - col_min) / safe_range

    values = normed.to_numpy()
    n = len(values)
    distances: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(values[i] - values[j]))
            distances.append(dist)

    return float(np.mean(distances)) if distances else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_diversity(
    plan: pd.DataFrame,
    proportion_col: str = "proportion",
    trait_columns: tuple[str, ...] = DEFAULT_TRAIT_COLUMNS,
) -> DiversityResult:
    """Compute ecological and functional diversity indices for a planting plan.

    All computations are performed on copies of the input; the original
    DataFrame is never mutated.

    Args:
        plan: DataFrame where each row represents a species in the planting
            plan.  Must contain a column with proportional abundances (areas,
            counts, or percentages — any non-negative numeric values are
            accepted and will be normalised internally).
        proportion_col: Name of the column holding proportional abundances.
            Defaults to ``"proportion"``.
        trait_columns: Tuple of column names to use as functional traits when
            computing functional diversity.  Columns absent from *plan* are
            silently ignored.  Defaults to :data:`DEFAULT_TRAIT_COLUMNS`.

    Returns:
        A :class:`DiversityResult` containing:

        - ``shannon_index`` — Shannon entropy H' (nats).
        - ``simpson_index`` — Simpson diversity D (0–1).
        - ``evenness`` — Pielou's evenness J' (0–1), or ``None`` for
          single-species plans.
        - ``functional_diversity`` — Mean pairwise trait distance (0–∞),
          or ``None`` when trait data is insufficient.
        - ``species_count`` — Count of species with non-zero proportion.
        - ``summary`` — Human-readable summary string.

    Raises:
        TypeError: If *plan* is not a :class:`pandas.DataFrame`.
        ValueError: If *plan* is empty, *proportion_col* is missing, any
            proportion is negative, or all proportions are zero.

    Example:
        >>> import pandas as pd
        >>> from src.species_diversity_scorer import compute_diversity
        >>> plan = pd.DataFrame({
        ...     "species_name": ["Teak", "Sengon", "Albizzia"],
        ...     "proportion": [0.5, 0.3, 0.2],
        ...     "growth_rate_m_yr": [1.5, 3.5, 2.0],
        ...     "carbon_seq_tc_ha_yr": [8.2, 13.2, 7.5],
        ... })
        >>> result = compute_diversity(plan)
        >>> print(f"Shannon H' = {result.shannon_index:.3f}")
        Shannon H' = 1.030
        >>> print(f"Simpson D  = {result.simpson_index:.3f}")
        Simpson D  = 0.620
        >>> print(f"Evenness J'= {result.evenness:.3f}")
        Evenness J'= 0.937
    """
    _validate_plan(plan, proportion_col)

    proportions = _normalise_proportions(plan[proportion_col])
    active_mask = proportions > _MIN_PROPORTION
    species_count = int(active_mask.sum())

    shannon = _shannon(proportions)
    simpson = _simpson(proportions)
    evenness = _pielou_evenness(shannon, species_count)
    func_div = _functional_diversity(plan[active_mask].reset_index(drop=True), trait_columns)

    summary_parts = [
        f"Species: {species_count}",
        f"Shannon H' = {shannon:.3f}",
        f"Simpson D  = {simpson:.3f}",
        f"Evenness J' = {evenness:.3f}" if evenness is not None else "Evenness J' = N/A",
        f"Functional diversity = {func_div:.3f}"
        if func_div is not None
        else "Functional diversity = N/A",
    ]
    summary = " | ".join(summary_parts)

    return DiversityResult(
        shannon_index=shannon,
        simpson_index=simpson,
        evenness=evenness,
        functional_diversity=func_div,
        species_count=species_count,
        summary=summary,
    )


def score_plan_diversity(
    plan: pd.DataFrame,
    proportion_col: str = "proportion",
    trait_columns: tuple[str, ...] = DEFAULT_TRAIT_COLUMNS,
) -> pd.DataFrame:
    """Return a one-row summary DataFrame with all diversity indices.

    Convenience wrapper around :func:`compute_diversity` that packages the
    result as a tidy DataFrame for easy integration with pandas pipelines.

    Args:
        plan: Planting plan DataFrame (see :func:`compute_diversity`).
        proportion_col: Name of the proportion column.
        trait_columns: Functional trait columns to use.

    Returns:
        Single-row DataFrame with columns:
        ``species_count``, ``shannon_index``, ``simpson_index``,
        ``evenness``, ``functional_diversity``.

    Raises:
        TypeError: Propagated from :func:`compute_diversity`.
        ValueError: Propagated from :func:`compute_diversity`.

    Example:
        >>> import pandas as pd
        >>> from src.species_diversity_scorer import score_plan_diversity
        >>> plan = pd.DataFrame({
        ...     "species_name": ["Teak", "Sengon"],
        ...     "proportion": [0.6, 0.4],
        ...     "growth_rate_m_yr": [1.5, 3.5],
        ... })
        >>> df = score_plan_diversity(plan)
        >>> print(df.columns.tolist())
        ['species_count', 'shannon_index', 'simpson_index', 'evenness', 'functional_diversity']
    """
    result = compute_diversity(plan, proportion_col=proportion_col, trait_columns=trait_columns)
    return pd.DataFrame(
        [
            {
                "species_count": result.species_count,
                "shannon_index": round(result.shannon_index, 6),
                "simpson_index": round(result.simpson_index, 6),
                "evenness": round(result.evenness, 6) if result.evenness is not None else None,
                "functional_diversity": (
                    round(result.functional_diversity, 6)
                    if result.functional_diversity is not None
                    else None
                ),
            }
        ]
    )
