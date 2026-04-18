"""
Species portfolio builder — compose an optimal diverse species mix for a site.

This module addresses a different planning question than
:mod:`src.site_match_scorer` (best individual species for a site) or
:mod:`src.species_diversity_scorer` (diversity of a given plan): *given a
candidate pool, which subset of species should be planted at a site, and in
what proportions, to maximise both site fit and ecological diversity?*

The core algorithm is a **greedy diversity-aware selector**:

1.  Score every candidate against the site using
    :func:`src.site_match_scorer.score_site_match`.
2.  Discard candidates whose ``site_match_score`` is below a configurable
    ``min_site_score`` floor.
3.  Seed the portfolio with the highest-scoring species.
4.  Iteratively add the candidate that maximises a combined objective:

        objective(s) = alpha * site_match_score(s)
                     + (1 - alpha) * mean_trait_distance(s, portfolio)

    where *mean_trait_distance* is the mean normalised Euclidean distance
    from species *s* to every species already in the portfolio (functional
    diversity).  ``alpha`` controls the fit-vs-diversity trade-off.
5.  Stop when ``portfolio_size`` species have been added (or the pool is
    exhausted).

Proportions are then assigned proportionally to each species's combined
objective value (normalised to sum to 1).

All public functions are pure: inputs are never modified and a fresh
DataFrame is returned.

Author: github.com/achmadnaufal
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.site_match_scorer import Site, score_site_match

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default number of species to include in the portfolio.
DEFAULT_PORTFOLIO_SIZE: int = 5

#: Default balance between site-fit (alpha) and diversity (1 - alpha).
DEFAULT_ALPHA: float = 0.6

#: Default minimum site-match score a candidate must reach to be considered.
DEFAULT_MIN_SITE_SCORE: float = 0.3

#: Default trait columns used for functional-distance computations.
DEFAULT_TRAIT_COLUMNS: tuple[str, ...] = (
    "growth_rate_m_yr",
    "carbon_seq_tc_ha_yr",
    "min_rainfall_mm",
    "max_rainfall_mm",
    "min_temp_c",
    "max_temp_c",
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioResult:
    """Immutable container for a built species portfolio.

    Attributes:
        portfolio: DataFrame with one row per selected species, containing
            at minimum ``species_name``, ``site_match_score``, a
            ``diversity_contribution`` column (mean trait distance to the
            portfolio at the moment of selection), ``objective``, and
            ``proportion``.  Proportions sum to 1.0.
        site: The :class:`~src.site_match_scorer.Site` used to build the
            portfolio.
        alpha: Fit-vs-diversity weight used during construction.
        mean_site_match: Weighted mean ``site_match_score`` across the
            portfolio (weighted by ``proportion``).
        summary: Human-readable single-line summary.
    """

    portfolio: pd.DataFrame
    site: Site
    alpha: float
    mean_site_match: float
    summary: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise_trait_matrix(
    df: pd.DataFrame,
    trait_columns: Sequence[str],
) -> np.ndarray:
    """Return a min-max normalised trait matrix for *df*.

    Only numeric columns present in *df* are used.  Missing values are
    imputed with the column mean.  Constant columns are kept but contribute
    zero to pairwise distance.

    Args:
        df: Species DataFrame.
        trait_columns: Candidate trait column names.

    Returns:
        2-D ``numpy`` array of shape ``(n_rows, n_used_traits)`` with values
        in ``[0, 1]``.  Shape ``(n_rows, 0)`` when no usable traits exist.
    """
    available = [
        c
        for c in trait_columns
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not available:
        return np.zeros((len(df), 0), dtype=float)

    traits = df[list(available)].copy()
    traits = traits.fillna(traits.mean(numeric_only=True))
    # Any remaining NaNs (column was entirely NaN) -> zero-fill
    traits = traits.fillna(0.0)

    col_min = traits.min()
    col_max = traits.max()
    col_range = (col_max - col_min).replace(0, 1)
    normed = (traits - col_min) / col_range
    return normed.to_numpy(dtype=float)


def _mean_distance_to_set(
    candidate_row: np.ndarray,
    selected_rows: np.ndarray,
) -> float:
    """Return mean Euclidean distance from *candidate_row* to each row in
    *selected_rows*.

    Args:
        candidate_row: 1-D array of normalised traits for one candidate.
        selected_rows: 2-D array of normalised traits for already-selected
            species; may be empty.

    Returns:
        Mean Euclidean distance, or 0.0 when *selected_rows* is empty or
        traits are unavailable (zero-width row).
    """
    if selected_rows.size == 0 or candidate_row.size == 0:
        return 0.0
    diffs = selected_rows - candidate_row
    dists = np.linalg.norm(diffs, axis=1)
    return float(dists.mean())


def _validate_inputs(
    species_df: pd.DataFrame,
    portfolio_size: int,
    alpha: float,
    min_site_score: float,
) -> None:
    """Validate user-supplied parameters.

    Args:
        species_df: Candidate species DataFrame.
        portfolio_size: Target number of species.
        alpha: Fit-vs-diversity weight.
        min_site_score: Minimum site-match cutoff.

    Raises:
        TypeError: If *species_df* is not a DataFrame.
        ValueError: If numeric parameters fall outside valid ranges.
    """
    if not isinstance(species_df, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame, got {type(species_df).__name__!r}."
        )
    if species_df.empty:
        raise ValueError("species_df is empty.")
    if not isinstance(portfolio_size, int) or portfolio_size < 1:
        raise ValueError(
            f"portfolio_size must be a positive integer, got {portfolio_size!r}."
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
    if not 0.0 <= min_site_score <= 1.0:
        raise ValueError(
            f"min_site_score must be in [0, 1], got {min_site_score}."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_portfolio(
    species_df: pd.DataFrame,
    site: Site,
    *,
    portfolio_size: int = DEFAULT_PORTFOLIO_SIZE,
    alpha: float = DEFAULT_ALPHA,
    min_site_score: float = DEFAULT_MIN_SITE_SCORE,
    trait_columns: Sequence[str] = DEFAULT_TRAIT_COLUMNS,
    site_match_kwargs: Optional[Mapping[str, object]] = None,
) -> PortfolioResult:
    """Build a diverse, site-matched species portfolio.

    Uses a greedy algorithm that trades off site fit against functional
    diversity.  Proportions are assigned proportionally to each selected
    species's combined objective value.

    Args:
        species_df: DataFrame with one row per candidate species.  Must
            contain the columns required by
            :func:`src.site_match_scorer.score_site_match` (i.e.
            ``species_name``, ``min_rainfall_mm``, ``max_rainfall_mm``,
            ``min_temp_c``, ``max_temp_c``, ``soil_type``).
        site: Target planting site.
        portfolio_size: Maximum number of species to include.  If fewer
            candidates pass ``min_site_score`` the portfolio will be
            smaller.  Must be a positive integer.
        alpha: Fit-vs-diversity weight.  ``alpha=1.0`` reduces to pure
            site-match ranking; ``alpha=0.0`` maximises diversity ignoring
            fit.  Must lie in ``[0, 1]``.
        min_site_score: Candidates with ``site_match_score`` below this
            threshold are excluded.  Must lie in ``[0, 1]``.
        trait_columns: Column names used to compute functional distance.
            Missing columns are silently skipped.
        site_match_kwargs: Extra keyword arguments forwarded to
            :func:`~src.site_match_scorer.score_site_match` (tolerances,
            soil partial score, weights).

    Returns:
        Immutable :class:`PortfolioResult`.  When no candidate meets
        ``min_site_score`` the returned ``portfolio`` DataFrame is empty
        and ``mean_site_match`` is ``0.0``.

    Raises:
        TypeError: If *species_df* is not a :class:`pandas.DataFrame`.
        ValueError: If any numeric parameter is out of range, or if
            *species_df* is empty or missing required columns (propagated
            from :func:`score_site_match`).

    Example:
        >>> import pandas as pd
        >>> from src.site_match_scorer import Site
        >>> from src.portfolio_builder import build_portfolio
        >>> species = pd.DataFrame({
        ...     "species_name": ["Teak", "Acacia", "Sengon", "Jabon", "Mahogany"],
        ...     "min_rainfall_mm": [1200, 1000, 1200, 1500, 1500],
        ...     "max_rainfall_mm": [2500, 3500, 3000, 4000, 3000],
        ...     "min_temp_c": [20, 18, 18, 22, 22],
        ...     "max_temp_c": [35, 38, 36, 38, 36],
        ...     "soil_type": ["loam", "sandy_loam", "loam", "clay", "clay_loam"],
        ...     "growth_rate_m_yr": [1.5, 2.5, 3.5, 2.8, 1.2],
        ...     "carbon_seq_tc_ha_yr": [8.2, 11.4, 13.2, 10.7, 9.1],
        ... })
        >>> site = Site(rainfall_mm=1800, temperature_c=27, soil_type="loam")
        >>> result = build_portfolio(species, site, portfolio_size=3)
        >>> len(result.portfolio) <= 3
        True
        >>> abs(result.portfolio["proportion"].sum() - 1.0) < 1e-6
        True
    """
    _validate_inputs(species_df, portfolio_size, alpha, min_site_score)

    extra = dict(site_match_kwargs) if site_match_kwargs else {}
    scored = score_site_match(species_df, site, **extra)

    eligible = scored[scored["site_match_score"] >= min_site_score].reset_index(drop=True)

    if eligible.empty:
        empty = scored.iloc[0:0].assign(
            diversity_contribution=pd.Series(dtype=float),
            objective=pd.Series(dtype=float),
            proportion=pd.Series(dtype=float),
            rank=pd.Series(dtype=int),
        )
        return PortfolioResult(
            portfolio=empty,
            site=site,
            alpha=float(alpha),
            mean_site_match=0.0,
            summary=(
                f"No candidates reached min_site_score={min_site_score:.2f} "
                f"for site {site.name or site.soil_type!r}."
            ),
        )

    trait_matrix = _normalise_trait_matrix(eligible, trait_columns)
    site_scores = eligible["site_match_score"].to_numpy(dtype=float)
    n_candidates = len(eligible)
    target_size = min(portfolio_size, n_candidates)

    selected_indices: list[int] = []
    diversity_contribs: list[float] = []
    objectives: list[float] = []

    remaining = set(range(n_candidates))

    # Seed: highest site-match score
    seed_idx = int(np.argmax(site_scores))
    selected_indices.append(seed_idx)
    diversity_contribs.append(0.0)
    objectives.append(float(alpha * site_scores[seed_idx]))
    remaining.discard(seed_idx)

    while len(selected_indices) < target_size and remaining:
        selected_rows = trait_matrix[selected_indices] if trait_matrix.size else np.zeros((0, 0))
        best_idx = -1
        best_obj = -np.inf
        best_div = 0.0

        for idx in remaining:
            cand_row = trait_matrix[idx] if trait_matrix.size else np.zeros(0)
            div = _mean_distance_to_set(cand_row, selected_rows)
            obj = alpha * site_scores[idx] + (1.0 - alpha) * div
            if obj > best_obj:
                best_obj = obj
                best_idx = idx
                best_div = div

        if best_idx < 0:
            break

        selected_indices.append(best_idx)
        diversity_contribs.append(float(best_div))
        objectives.append(float(best_obj))
        remaining.discard(best_idx)

    portfolio = eligible.iloc[selected_indices].reset_index(drop=True).copy()
    portfolio = portfolio.assign(
        diversity_contribution=np.round(diversity_contribs, 4),
        objective=np.round(objectives, 4),
    )

    obj_values = np.asarray(objectives, dtype=float)
    total = obj_values.sum()
    if total > 0:
        proportions = obj_values / total
    else:
        # Degenerate: all objective values are zero -> equal proportions
        proportions = np.full(len(obj_values), 1.0 / len(obj_values))

    portfolio = portfolio.assign(
        proportion=np.round(proportions, 4),
        rank=range(1, len(portfolio) + 1),
    )

    weighted_mean_fit = float(
        np.sum(portfolio["site_match_score"].to_numpy(dtype=float) * proportions)
    )

    summary = (
        f"Portfolio of {len(portfolio)} species for "
        f"{site.name or site.soil_type!r} "
        f"(alpha={alpha:.2f}, weighted site-fit={weighted_mean_fit:.3f})"
    )

    return PortfolioResult(
        portfolio=portfolio,
        site=site,
        alpha=float(alpha),
        mean_site_match=weighted_mean_fit,
        summary=summary,
    )


def compare_portfolios(
    species_df: pd.DataFrame,
    site: Site,
    alphas: Sequence[float] = (0.0, 0.3, 0.6, 1.0),
    **kwargs,
) -> pd.DataFrame:
    """Compare portfolios built with different fit-vs-diversity weights.

    Convenience helper that runs :func:`build_portfolio` across several
    *alpha* values and returns a tidy comparison DataFrame.  Useful for
    sensitivity analysis.

    Args:
        species_df: Candidate species DataFrame.
        site: Target planting site.
        alphas: Iterable of alpha values to try.  Each must lie in [0, 1].
        **kwargs: Additional keyword arguments forwarded to
            :func:`build_portfolio`.

    Returns:
        DataFrame with columns ``alpha``, ``n_species``, ``weighted_site_fit``,
        ``mean_diversity_contribution``, and ``species_names`` (comma-joined).

    Raises:
        TypeError: Propagated from :func:`build_portfolio`.
        ValueError: If *alphas* is empty or any alpha is out of range.

    Example:
        >>> # doctest: +SKIP
        >>> comp = compare_portfolios(species, site, alphas=[0.0, 0.5, 1.0])
        >>> list(comp["alpha"])
        [0.0, 0.5, 1.0]
    """
    alpha_list = list(alphas)
    if not alpha_list:
        raise ValueError("alphas must contain at least one value.")
    for a in alpha_list:
        if not 0.0 <= a <= 1.0:
            raise ValueError(f"Each alpha must be in [0, 1], got {a}.")

    rows: list[dict] = []
    for a in alpha_list:
        result = build_portfolio(species_df, site, alpha=a, **kwargs)
        pf = result.portfolio
        if pf.empty:
            rows.append(
                {
                    "alpha": float(a),
                    "n_species": 0,
                    "weighted_site_fit": 0.0,
                    "mean_diversity_contribution": 0.0,
                    "species_names": "",
                }
            )
            continue
        rows.append(
            {
                "alpha": float(a),
                "n_species": len(pf),
                "weighted_site_fit": round(result.mean_site_match, 4),
                "mean_diversity_contribution": round(
                    float(pf["diversity_contribution"].mean()), 4
                ),
                "species_names": ", ".join(pf["species_name"].astype(str).tolist()),
            }
        )
    return pd.DataFrame(rows)
