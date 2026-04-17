"""
Site-match scorer for tree species selection.

This module evaluates how well each candidate species fits a given planting
*site* characterised by its mean annual rainfall, mean annual temperature,
and soil type.  It complements :mod:`src.main` (composite suitability score
computed across the whole dataset) by providing a *site-conditional* score:
the same species can score very differently depending on the target site.

For each species, three site-specific sub-scores are computed:

* **Rainfall match** -- 1.0 if site rainfall lies inside the species
  ``[min_rainfall_mm, max_rainfall_mm]`` envelope; decays linearly to 0 as
  the site rainfall moves outside the envelope by up to a configurable
  tolerance margin.
* **Temperature match** -- analogous to rainfall using
  ``[min_temp_c, max_temp_c]``.
* **Soil match** -- 1.0 for an exact soil-type match, a configurable partial
  score for "compatible" soil types (e.g. ``loam`` vs ``clay_loam``), 0.0
  otherwise.

The three sub-scores are combined with configurable weights (default
0.4 / 0.4 / 0.2) into an overall ``site_match_score`` in the range [0, 1].

All public functions are pure: inputs are never modified and a fresh
DataFrame is returned.

Author: github.com/achmadnaufal
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from src.main import VALID_SOIL_TYPES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default tolerance (mm) outside the rainfall envelope before score hits 0.
DEFAULT_RAINFALL_TOLERANCE_MM: float = 500.0

#: Default tolerance (degrees C) outside the temperature envelope before
#: score hits 0.
DEFAULT_TEMP_TOLERANCE_C: float = 5.0

#: Default partial-match score for "related" soil types.
DEFAULT_SOIL_PARTIAL_SCORE: float = 0.5

#: Default sub-score weights for the composite ``site_match_score``.
DEFAULT_WEIGHTS: Mapping[str, float] = {
    "rainfall": 0.4,
    "temperature": 0.4,
    "soil": 0.2,
}

#: Soil-type compatibility groups -- members of the same group score the
#: ``soil_partial_score`` against each other.
_SOIL_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"loam", "clay_loam", "sandy_loam", "silt"}),
    frozenset({"sandy", "sandy_loam"}),
    frozenset({"clay", "clay_loam"}),
    frozenset({"peat", "silt"}),
)


# ---------------------------------------------------------------------------
# Site definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Site:
    """Immutable description of a candidate planting site.

    Attributes:
        rainfall_mm: Mean annual rainfall at the site, in millimetres.
            Must be non-negative.
        temperature_c: Mean annual temperature at the site, in degrees C.
        soil_type: Dominant soil type at the site.  Must be one of
            :data:`src.main.VALID_SOIL_TYPES`.
        name: Optional human-readable site label used in summaries.

    Raises:
        ValueError: If ``rainfall_mm`` is negative or ``soil_type`` is not
            a recognised value.
    """

    rainfall_mm: float
    temperature_c: float
    soil_type: str
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.rainfall_mm < 0:
            raise ValueError(
                f"rainfall_mm must be >= 0, got {self.rainfall_mm}."
            )
        soil_lower = self.soil_type.lower()
        if soil_lower not in VALID_SOIL_TYPES:
            raise ValueError(
                f"Invalid soil_type {self.soil_type!r}. "
                f"Must be one of {sorted(VALID_SOIL_TYPES)}."
            )
        # Frozen dataclass: bypass to normalise soil_type case
        object.__setattr__(self, "soil_type", soil_lower)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _envelope_score(
    value: float,
    minimum: float,
    maximum: float,
    tolerance: float,
) -> float:
    """Return a [0, 1] score for how well *value* fits ``[minimum, maximum]``.

    The score is 1.0 when *value* lies inside the envelope, decreases
    linearly to 0.0 once *value* is *tolerance* units outside the envelope,
    and is clamped at 0.0 beyond that.

    Args:
        value: Observed value at the site.
        minimum: Lower bound of the species's tolerance envelope.
        maximum: Upper bound of the species's tolerance envelope.
        tolerance: Distance outside the envelope at which the score reaches 0.

    Returns:
        Match score in the closed interval [0.0, 1.0].
    """
    if tolerance <= 0:
        return 1.0 if minimum <= value <= maximum else 0.0
    if minimum <= value <= maximum:
        return 1.0
    if value < minimum:
        deficit = minimum - value
    else:
        deficit = value - maximum
    return float(max(0.0, 1.0 - deficit / tolerance))


def _soils_compatible(a: str, b: str) -> bool:
    """Return True when *a* and *b* share at least one compatibility group."""
    return any(a in g and b in g for g in _SOIL_GROUPS)


def _soil_score(site_soil: str, species_soil: str, partial_score: float) -> float:
    """Compute the soil sub-score for one species against *site_soil*.

    Args:
        site_soil: Soil type at the site (already lower-cased).
        species_soil: Preferred soil type for the species.
        partial_score: Score awarded when soils are compatible but not equal.

    Returns:
        1.0 for exact match, *partial_score* for compatible soils,
        0.0 otherwise.
    """
    if not isinstance(species_soil, str):
        return 0.0
    species_lower = species_soil.strip().lower()
    if species_lower == site_soil:
        return 1.0
    if _soils_compatible(site_soil, species_lower):
        return float(partial_score)
    return 0.0


def _validate_species_df(species_df: pd.DataFrame) -> None:
    """Validate the species DataFrame at the system boundary.

    Args:
        species_df: Candidate species DataFrame.

    Raises:
        TypeError: If ``species_df`` is not a :class:`pandas.DataFrame`.
        ValueError: If ``species_df`` is empty or required columns are missing.
    """
    if not isinstance(species_df, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame, got {type(species_df).__name__!r}."
        )
    if species_df.empty:
        raise ValueError("species_df is empty.")
    required = {
        "species_name",
        "min_rainfall_mm",
        "max_rainfall_mm",
        "min_temp_c",
        "max_temp_c",
        "soil_type",
    }
    missing = required - set(species_df.columns)
    if missing:
        raise ValueError(
            f"species_df is missing required columns: {sorted(missing)}."
        )


def _resolve_weights(weights: Optional[Mapping[str, float]]) -> Mapping[str, float]:
    """Return validated weights dict (defaults applied when ``None``).

    Args:
        weights: Optional override mapping with keys ``rainfall``,
            ``temperature``, ``soil``.

    Returns:
        Mapping with all three keys present.

    Raises:
        ValueError: If supplied weights contain negatives or sum to 0.
    """
    if weights is None:
        return DEFAULT_WEIGHTS
    resolved = {
        "rainfall": float(weights.get("rainfall", DEFAULT_WEIGHTS["rainfall"])),
        "temperature": float(weights.get("temperature", DEFAULT_WEIGHTS["temperature"])),
        "soil": float(weights.get("soil", DEFAULT_WEIGHTS["soil"])),
    }
    if any(v < 0 for v in resolved.values()):
        raise ValueError(f"Weights must be non-negative, got {resolved}.")
    if sum(resolved.values()) == 0:
        raise ValueError("Weights sum to zero; at least one must be positive.")
    return resolved


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_site_match(
    species_df: pd.DataFrame,
    site: Site,
    *,
    rainfall_tolerance_mm: float = DEFAULT_RAINFALL_TOLERANCE_MM,
    temp_tolerance_c: float = DEFAULT_TEMP_TOLERANCE_C,
    soil_partial_score: float = DEFAULT_SOIL_PARTIAL_SCORE,
    weights: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    """Score every species in *species_df* against the supplied *site*.

    For each species, computes:

    * ``rainfall_match`` -- how well the site rainfall fits the species
      envelope ``[min_rainfall_mm, max_rainfall_mm]``.
    * ``temperature_match`` -- how well the site temperature fits the
      species envelope ``[min_temp_c, max_temp_c]``.
    * ``soil_match`` -- exact / compatible / unrelated soil match.
    * ``site_match_score`` -- weighted combination in [0, 1].

    The original *species_df* is never mutated; a new DataFrame is returned.

    Args:
        species_df: DataFrame with one row per candidate species.  Must
            contain the columns ``species_name``, ``min_rainfall_mm``,
            ``max_rainfall_mm``, ``min_temp_c``, ``max_temp_c``,
            ``soil_type``.
        site: :class:`Site` describing the planting location.
        rainfall_tolerance_mm: Distance (mm) outside the rainfall envelope
            beyond which ``rainfall_match`` is 0.  Must be > 0.
        temp_tolerance_c: Distance (deg C) outside the temperature envelope
            beyond which ``temperature_match`` is 0.  Must be > 0.
        soil_partial_score: Score for compatible-but-not-identical soil
            types.  Must be in [0, 1].
        weights: Optional mapping of sub-score weights with keys
            ``rainfall``, ``temperature``, ``soil``.  Defaults to
            :data:`DEFAULT_WEIGHTS`.

    Returns:
        New DataFrame -- a copy of *species_df* with four added columns:
        ``rainfall_match``, ``temperature_match``, ``soil_match``,
        ``site_match_score``.  Rows are not reordered.

    Raises:
        TypeError: If *species_df* is not a :class:`pandas.DataFrame`.
        ValueError: If *species_df* is empty, required columns are missing,
            tolerances are non-positive, or *soil_partial_score* is out of
            range.

    Example:
        >>> import pandas as pd
        >>> from src.site_match_scorer import Site, score_site_match
        >>> species = pd.DataFrame({
        ...     "species_name": ["Teak", "Scots Pine"],
        ...     "min_rainfall_mm": [1200, 300],
        ...     "max_rainfall_mm": [2500, 800],
        ...     "min_temp_c": [20, -5],
        ...     "max_temp_c": [35, 25],
        ...     "soil_type": ["loam", "sandy"],
        ... })
        >>> site = Site(rainfall_mm=1800, temperature_c=27, soil_type="loam")
        >>> result = score_site_match(species, site)
        >>> result.iloc[0]["site_match_score"] > result.iloc[1]["site_match_score"]
        True
    """
    _validate_species_df(species_df)
    if rainfall_tolerance_mm <= 0:
        raise ValueError(
            f"rainfall_tolerance_mm must be > 0, got {rainfall_tolerance_mm}."
        )
    if temp_tolerance_c <= 0:
        raise ValueError(
            f"temp_tolerance_c must be > 0, got {temp_tolerance_c}."
        )
    if not 0.0 <= soil_partial_score <= 1.0:
        raise ValueError(
            f"soil_partial_score must be in [0, 1], got {soil_partial_score}."
        )

    resolved_weights = _resolve_weights(weights)
    weight_sum = sum(resolved_weights.values())

    # Work on a copy to preserve immutability of the caller's data.
    result = species_df.copy()

    rainfall_scores = [
        _envelope_score(
            site.rainfall_mm,
            float(row["min_rainfall_mm"]),
            float(row["max_rainfall_mm"]),
            rainfall_tolerance_mm,
        )
        for _, row in result.iterrows()
    ]
    temp_scores = [
        _envelope_score(
            site.temperature_c,
            float(row["min_temp_c"]),
            float(row["max_temp_c"]),
            temp_tolerance_c,
        )
        for _, row in result.iterrows()
    ]
    soil_scores = [
        _soil_score(site.soil_type, row["soil_type"], soil_partial_score)
        for _, row in result.iterrows()
    ]

    rainfall_arr = np.asarray(rainfall_scores, dtype=float)
    temp_arr = np.asarray(temp_scores, dtype=float)
    soil_arr = np.asarray(soil_scores, dtype=float)

    composite = (
        resolved_weights["rainfall"] * rainfall_arr
        + resolved_weights["temperature"] * temp_arr
        + resolved_weights["soil"] * soil_arr
    ) / weight_sum

    return result.assign(
        rainfall_match=np.round(rainfall_arr, 4),
        temperature_match=np.round(temp_arr, 4),
        soil_match=np.round(soil_arr, 4),
        site_match_score=np.round(composite, 4),
    )


def recommend_for_site(
    species_df: pd.DataFrame,
    site: Site,
    *,
    top_n: int = 5,
    min_score: float = 0.0,
    **kwargs,
) -> pd.DataFrame:
    """Return the *top_n* species best matched to *site*, sorted descending.

    Convenience wrapper around :func:`score_site_match` that filters by a
    minimum acceptable score and truncates to a top-N list, sorted by
    ``site_match_score`` descending.

    Args:
        species_df: Candidate species DataFrame (see :func:`score_site_match`).
        site: Target :class:`Site`.
        top_n: Maximum number of species to return.  Must be a positive int.
        min_score: Minimum ``site_match_score`` required (in [0, 1]).
            Species below this threshold are dropped.  Defaults to 0.0.
        **kwargs: Additional keyword arguments forwarded to
            :func:`score_site_match` (tolerances, partial score, weights).

    Returns:
        New DataFrame ranked by ``site_match_score`` descending, with at
        most *top_n* rows and an added ``rank`` column starting at 1.

    Raises:
        TypeError: Propagated from :func:`score_site_match`.
        ValueError: If *top_n* is not a positive integer or *min_score* is
            outside [0, 1]; otherwise propagated.

    Example:
        >>> import pandas as pd
        >>> from src.site_match_scorer import Site, recommend_for_site
        >>> species = pd.DataFrame({
        ...     "species_name": ["Teak", "Scots Pine", "Acacia"],
        ...     "min_rainfall_mm": [1200, 300, 1000],
        ...     "max_rainfall_mm": [2500, 800, 3500],
        ...     "min_temp_c": [20, -5, 18],
        ...     "max_temp_c": [35, 25, 38],
        ...     "soil_type": ["loam", "sandy", "sandy_loam"],
        ... })
        >>> site = Site(rainfall_mm=1800, temperature_c=27, soil_type="loam")
        >>> top = recommend_for_site(species, site, top_n=2)
        >>> list(top["rank"])
        [1, 2]
    """
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError(f"top_n must be a positive integer, got {top_n!r}.")
    if not 0.0 <= min_score <= 1.0:
        raise ValueError(f"min_score must be in [0, 1], got {min_score}.")

    scored = score_site_match(species_df, site, **kwargs)
    filtered = scored[scored["site_match_score"] >= min_score]
    ranked = (
        filtered.sort_values("site_match_score", ascending=False)
        .reset_index(drop=True)
        .head(top_n)
        .reset_index(drop=True)
    )
    return ranked.assign(rank=range(1, len(ranked) + 1))
