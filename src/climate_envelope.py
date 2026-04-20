"""
Climate envelope matching for tree species using WorldClim BIO variables.

This module implements a bioclimatic envelope model (a simplified BIOCLIM-style
approach) that evaluates how well a candidate planting site matches each
species' known climatic tolerance range, expressed in terms of the standard
WorldClim bioclimatic variables (BIO1, BIO5, BIO6, BIO12, BIO14, BIO17).

References
----------
- Booth, T.H. et al. (2014). "BIOCLIM: the first species distribution
  modelling package, its early applications and relevance to most current
  MaxEnt studies." Diversity and Distributions 20(1): 1-9.
- Fick, S.E. & Hijmans, R.J. (2017). "WorldClim 2: new 1-km spatial
  resolution climate surfaces for global land areas." International Journal
  of Climatology 37(12): 4302-4315.

Author: github.com/achmadnaufal
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# WorldClim BIO variable metadata
# ---------------------------------------------------------------------------
#
# Only the six most commonly used WorldClim variables for species distribution
# modelling are supported. Each key maps to a human-readable description and
# the physical unit.
BIOCLIM_VARIABLES: Dict[str, Tuple[str, str]] = {
    "bio1": ("Annual Mean Temperature", "C"),
    "bio5": ("Max Temperature of Warmest Month", "C"),
    "bio6": ("Min Temperature of Coldest Month", "C"),
    "bio12": ("Annual Precipitation", "mm"),
    "bio14": ("Precipitation of Driest Month", "mm"),
    "bio17": ("Precipitation of Driest Quarter", "mm"),
}

# Default weight for each BIO variable when computing the composite match.
# Weights must sum to 1.0 and emphasise annual mean temperature and annual
# precipitation, which are the two strongest predictors in most tropical and
# subtropical SDM studies.
DEFAULT_BIO_WEIGHTS: Dict[str, float] = {
    "bio1": 0.25,
    "bio5": 0.15,
    "bio6": 0.15,
    "bio12": 0.25,
    "bio14": 0.10,
    "bio17": 0.10,
}

# Match category thresholds (percent score out of 100).
_MATCH_CATEGORIES: List[Tuple[float, str]] = [
    (85.0, "excellent"),
    (65.0, "good"),
    (40.0, "marginal"),
    (0.0, "unsuitable"),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BioclimRange:
    """Immutable bioclimatic tolerance range for a single BIO variable.

    Parameters
    ----------
    variable:
        The WorldClim variable key (e.g. ``"bio1"``).
    low:
        Lower bound of the species' tolerance range (inclusive).
    high:
        Upper bound of the species' tolerance range (inclusive).
    """

    variable: str
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.variable not in BIOCLIM_VARIABLES:
            raise ValueError(
                f"Unknown BIO variable {self.variable!r}. "
                f"Must be one of {sorted(BIOCLIM_VARIABLES)}."
            )
        if self.low > self.high:
            raise ValueError(
                f"{self.variable}: low ({self.low}) must be <= high ({self.high})."
            )

    @property
    def width(self) -> float:
        """Return the width (high - low) of the tolerance range."""
        return self.high - self.low

    def score(self, value: float) -> float:
        """Score a site's observed *value* against this tolerance range.

        The score is 1.0 when the value is inside the range, and decays
        linearly to 0.0 over a buffer equal to 25 % of the range width
        outside the bounds. Values beyond that buffer score 0.0.

        Parameters
        ----------
        value:
            Observed site value for this BIO variable.

        Returns
        -------
        float
            Score in ``[0.0, 1.0]``.
        """
        if self.low <= value <= self.high:
            return 1.0
        # Use at least 1.0 as the buffer to avoid division-by-zero on
        # degenerate (width=0) ranges, which may occur for highly specialised
        # species.
        buffer = max(self.width * 0.25, 1.0)
        if value < self.low:
            deficit = self.low - value
        else:
            deficit = value - self.high
        if deficit >= buffer:
            return 0.0
        return max(0.0, 1.0 - deficit / buffer)


@dataclass(frozen=True)
class SpeciesEnvelope:
    """Bioclimatic envelope for a single species.

    Parameters
    ----------
    species_name:
        Common name of the species.
    scientific_name:
        Scientific (Latin) binomial name.
    ranges:
        Mapping of BIO variable key to :class:`BioclimRange`.
    """

    species_name: str
    scientific_name: str
    ranges: Dict[str, BioclimRange] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.species_name or not self.species_name.strip():
            raise ValueError("species_name must be a non-empty string.")
        if not self.scientific_name or not self.scientific_name.strip():
            raise ValueError("scientific_name must be a non-empty string.")
        for key, rng in self.ranges.items():
            if key != rng.variable:
                raise ValueError(
                    f"Envelope key {key!r} does not match range variable {rng.variable!r}."
                )


@dataclass(frozen=True)
class EnvelopeMatchResult:
    """Result of matching a site against one species envelope.

    Parameters
    ----------
    species_name:
        Common name of the species.
    scientific_name:
        Scientific (Latin) binomial name.
    match_score:
        Composite match score, 0-100.
    category:
        One of ``excellent``, ``good``, ``marginal``, ``unsuitable``.
    per_variable_scores:
        Mapping of BIO variable key to its individual 0-1 score.
    """

    species_name: str
    scientific_name: str
    match_score: float
    category: str
    per_variable_scores: Dict[str, float]


# ---------------------------------------------------------------------------
# Core matcher
# ---------------------------------------------------------------------------


class ClimateEnvelopeMatcher:
    """Match planting sites to species via bioclimatic envelopes.

    The matcher evaluates a site's WorldClim BIO variables against each
    species' known tolerance ranges and returns a ranked list of candidate
    species ordered by composite match score.

    Parameters
    ----------
    weights:
        Optional override of the per-variable weights. Must contain the same
        six keys as :data:`DEFAULT_BIO_WEIGHTS` and sum to 1.0 (within a
        tolerance of 1e-6).

    Examples
    --------
    >>> matcher = ClimateEnvelopeMatcher()
    >>> envelopes = matcher.load_envelopes("sample_data/climate_envelope_samples.csv")
    >>> site = {"bio1": 26.5, "bio5": 33.0, "bio6": 21.0,
    ...         "bio12": 2100, "bio14": 45, "bio17": 180}
    >>> ranked = matcher.rank_species(site, envelopes, top_n=3)
    >>> [(r.species_name, r.category) for r in ranked]  # doctest: +SKIP
    [('Teak', 'excellent'), ('Sengon', 'good'), ('Mahogany', 'good')]
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self._weights: Dict[str, float] = self._validate_weights(weights or DEFAULT_BIO_WEIGHTS)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_weights(weights: Dict[str, float]) -> Dict[str, float]:
        """Return a validated copy of *weights*.

        Parameters
        ----------
        weights:
            Mapping of BIO variable key to weight.

        Returns
        -------
        dict
            New dict with validated, copied entries.

        Raises
        ------
        ValueError
            If the keys differ from :data:`BIOCLIM_VARIABLES` or the weights
            do not sum to 1.0 (tolerance 1e-6).
        """
        missing = set(BIOCLIM_VARIABLES) - set(weights)
        extra = set(weights) - set(BIOCLIM_VARIABLES)
        if missing or extra:
            raise ValueError(
                f"weights must contain exactly these keys: {sorted(BIOCLIM_VARIABLES)}. "
                f"Missing={sorted(missing)}, extra={sorted(extra)}."
            )
        for k, v in weights.items():
            if v < 0:
                raise ValueError(f"weights[{k!r}] must be >= 0, got {v}.")
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {total:.6f}.")
        return dict(weights)

    @staticmethod
    def _validate_site(site: Dict[str, float]) -> Dict[str, float]:
        """Return a validated copy of a site observation dict.

        Parameters
        ----------
        site:
            Mapping of BIO variable key to observed value.

        Returns
        -------
        dict
            New dict with validated (float) values for the six BIO variables.

        Raises
        ------
        TypeError
            If *site* is not a dict.
        ValueError
            If a required BIO variable is missing, has a non-numeric value,
            or has a physically impossible value (negative precipitation).
        """
        if not isinstance(site, dict):
            raise TypeError(f"site must be a dict, got {type(site).__name__!r}.")
        cleaned: Dict[str, float] = {}
        for key in BIOCLIM_VARIABLES:
            if key not in site:
                raise ValueError(f"site is missing required BIO variable {key!r}.")
            try:
                val = float(site[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"site[{key!r}] must be numeric, got {site[key]!r}."
                ) from exc
            if key in ("bio12", "bio14", "bio17") and val < 0:
                raise ValueError(
                    f"site[{key!r}] is a precipitation variable and cannot be negative ({val})."
                )
            cleaned[key] = val
        return cleaned

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_envelopes(filepath: str) -> List[SpeciesEnvelope]:
        """Load species envelopes from a CSV file.

        The CSV must have a header row containing at least the columns:
        ``species_name``, ``scientific_name``, and one pair of ``<bio>_low``
        / ``<bio>_high`` columns per BIO variable present.

        Parameters
        ----------
        filepath:
            Path to the CSV file.

        Returns
        -------
        list of SpeciesEnvelope
            Parsed envelopes in file order.

        Raises
        ------
        FileNotFoundError
            If *filepath* does not exist.
        ValueError
            If the file is empty or missing required columns.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Envelope file not found: {filepath}")
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"Envelope file is empty: {filepath}")
        for required in ("species_name", "scientific_name"):
            if required not in df.columns:
                raise ValueError(f"Envelope file is missing required column {required!r}.")

        envelopes: List[SpeciesEnvelope] = []
        for _, row in df.iterrows():
            ranges: Dict[str, BioclimRange] = {}
            for bio_key in BIOCLIM_VARIABLES:
                low_col = f"{bio_key}_low"
                high_col = f"{bio_key}_high"
                if low_col in df.columns and high_col in df.columns:
                    low_val = row[low_col]
                    high_val = row[high_col]
                    if pd.isna(low_val) or pd.isna(high_val):
                        continue
                    ranges = {
                        **ranges,
                        bio_key: BioclimRange(
                            variable=bio_key,
                            low=float(low_val),
                            high=float(high_val),
                        ),
                    }
            envelope = SpeciesEnvelope(
                species_name=str(row["species_name"]).strip(),
                scientific_name=str(row["scientific_name"]).strip(),
                ranges=ranges,
            )
            envelopes = [*envelopes, envelope]
        return envelopes

    # ------------------------------------------------------------------
    # Scoring & ranking
    # ------------------------------------------------------------------

    def match_site(
        self, site: Dict[str, float], envelope: SpeciesEnvelope
    ) -> EnvelopeMatchResult:
        """Compute the composite envelope match score for a single species.

        Parameters
        ----------
        site:
            Mapping of BIO variable key to observed site value (six keys).
        envelope:
            The species envelope to evaluate.

        Returns
        -------
        EnvelopeMatchResult
            Immutable result object with per-variable scores and category.

        Raises
        ------
        ValueError
            If *site* is missing required BIO variables or has negative
            precipitation values.
        """
        validated_site = self._validate_site(site)
        per_variable: Dict[str, float] = {}
        total_score = 0.0
        total_weight = 0.0
        for bio_key, weight in self._weights.items():
            rng = envelope.ranges.get(bio_key)
            if rng is None:
                continue
            score = rng.score(validated_site[bio_key])
            per_variable = {**per_variable, bio_key: score}
            total_score += weight * score
            total_weight += weight

        if total_weight == 0.0:
            # Envelope has no usable BIO variables; return an explicit
            # unsuitable result rather than dividing by zero.
            composite = 0.0
        else:
            composite = (total_score / total_weight) * 100.0

        rounded = round(composite, 2)
        return EnvelopeMatchResult(
            species_name=envelope.species_name,
            scientific_name=envelope.scientific_name,
            match_score=rounded,
            category=categorise_score(rounded),
            per_variable_scores=per_variable,
        )

    def rank_species(
        self,
        site: Dict[str, float],
        envelopes: List[SpeciesEnvelope],
        top_n: Optional[int] = None,
    ) -> List[EnvelopeMatchResult]:
        """Rank species envelopes by match score (highest first).

        Parameters
        ----------
        site:
            Mapping of BIO variable key to observed site value.
        envelopes:
            Iterable of :class:`SpeciesEnvelope` instances.
        top_n:
            If given, return only the top *top_n* matches.

        Returns
        -------
        list of EnvelopeMatchResult
            New list, sorted descending by match score.

        Raises
        ------
        ValueError
            If *envelopes* is empty or *top_n* is not a positive integer.
        """
        if not envelopes:
            raise ValueError("envelopes list must not be empty.")
        if top_n is not None and (not isinstance(top_n, int) or top_n < 1):
            raise ValueError(f"top_n must be a positive integer, got {top_n!r}.")

        results = [self.match_site(site, env) for env in envelopes]
        ranked = sorted(results, key=lambda r: r.match_score, reverse=True)
        if top_n is not None:
            ranked = ranked[:top_n]
        return ranked

    def to_dataframe(self, results: List[EnvelopeMatchResult]) -> pd.DataFrame:
        """Convert a list of match results to a tidy DataFrame.

        Parameters
        ----------
        results:
            List of :class:`EnvelopeMatchResult` objects.

        Returns
        -------
        pd.DataFrame
            Columns: ``rank``, ``species_name``, ``scientific_name``,
            ``match_score``, ``category``, and one column per BIO variable
            with its individual score.
        """
        rows: List[Dict[str, object]] = []
        for idx, r in enumerate(results, start=1):
            row: Dict[str, object] = {
                "rank": idx,
                "species_name": r.species_name,
                "scientific_name": r.scientific_name,
                "match_score": r.match_score,
                "category": r.category,
            }
            for bio_key in BIOCLIM_VARIABLES:
                row = {**row, f"{bio_key}_score": r.per_variable_scores.get(bio_key)}
            rows = [*rows, row]
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def categorise_score(score: float) -> str:
    """Return the descriptive category for a composite *score* (0-100).

    Parameters
    ----------
    score:
        Composite match score, 0-100.

    Returns
    -------
    str
        One of ``excellent`` (>=85), ``good`` (>=65), ``marginal`` (>=40),
        or ``unsuitable`` (<40).

    Raises
    ------
    ValueError
        If *score* is outside the [0, 100] range.
    """
    if score < 0 or score > 100:
        raise ValueError(f"score must be in [0, 100], got {score}.")
    for threshold, label in _MATCH_CATEGORIES:
        if score >= threshold:
            return label
    return "unsuitable"


def widen_envelope(envelope: SpeciesEnvelope, factor: float) -> SpeciesEnvelope:
    """Return a new envelope with each range widened by *factor* on each side.

    Used to model climate-change buffer zones or to reflect taxonomic
    uncertainty. Returns a fresh object; the input envelope is not mutated.

    Parameters
    ----------
    envelope:
        Input species envelope.
    factor:
        Fractional widening (e.g. ``0.1`` expands each bound by 10 % of the
        range width). Must be >= 0.

    Returns
    -------
    SpeciesEnvelope
        New envelope with widened ranges.

    Raises
    ------
    ValueError
        If *factor* is negative.
    """
    if factor < 0:
        raise ValueError(f"factor must be >= 0, got {factor}.")
    widened: Dict[str, BioclimRange] = {}
    for key, rng in envelope.ranges.items():
        delta = rng.width * factor
        widened = {
            **widened,
            key: BioclimRange(variable=key, low=rng.low - delta, high=rng.high + delta),
        }
    return replace(envelope, ranges=widened)
