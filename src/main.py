"""
Decision support tool for optimal tree species selection by climate and soil zone.

This module provides the :class:`SpeciesSelector` class, which loads a species
dataset (CSV or Excel), validates it, filters candidates by environmental
criteria, scores them, and returns ranked recommendations.

Author: github.com/achmadnaufal
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List

# ---------------------------------------------------------------------------
# Valid domain values used for input validation
# ---------------------------------------------------------------------------
VALID_CLIMATE_ZONES: frozenset = frozenset({"tropical", "subtropical", "temperate", "boreal", "arid"})
VALID_SOIL_TYPES: frozenset = frozenset(
    {"loam", "clay", "sandy", "clay_loam", "sandy_loam", "silt", "peat"}
)

# Scoring weights (must sum to 1.0)
_WEIGHT_CARBON = 0.40
_WEIGHT_GROWTH = 0.30
_WEIGHT_NATIVE = 0.15
_WEIGHT_AGROFORESTRY = 0.10
_WEIGHT_DROUGHT = 0.05


class SpeciesSelector:
    """Climate-based tree species selector.

    Loads a species dataset and provides methods to filter candidates by
    environmental criteria (climate zone, rainfall, soil type) and rank them
    by a composite suitability score.

    Parameters
    ----------
    config:
        Optional dictionary of configuration overrides. Supported keys:

        * ``score_weights`` – dict with keys ``carbon``, ``growth``,
          ``native``, ``agroforestry``, ``drought`` mapping to float weights
          (must sum to 1.0).

    Examples
    --------
    >>> selector = SpeciesSelector()
    >>> df = selector.load_data("demo/sample_data.csv")
    >>> filtered = selector.filter(df, climate_zone="tropical", min_rainfall_mm=1200)
    >>> ranked = selector.rank(filtered)
    >>> print(ranked[["species_name", "suitability_score"]].head())
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the selector.

        Parameters
        ----------
        config:
            Optional configuration dictionary.
        """
        self.config: Dict[str, Any] = config or {}

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load species data from a CSV or Excel file.

        Parameters
        ----------
        filepath:
            Path to the input file.  Accepts ``*.csv``, ``*.xlsx``, and
            ``*.xls`` formats.

        Returns
        -------
        pd.DataFrame
            Raw (un-preprocessed) data loaded from *filepath*.

        Raises
        ------
        FileNotFoundError
            If *filepath* does not exist.
        ValueError
            If the file extension is not supported.
        """
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        if p.suffix in (".xlsx", ".xls"):
            return pd.read_excel(filepath)
        if p.suffix == ".csv":
            return pd.read_csv(filepath)
        raise ValueError(f"Unsupported file format: {p.suffix!r}. Use .csv, .xlsx, or .xls.")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate that *df* meets minimum structural requirements.

        Parameters
        ----------
        df:
            DataFrame to validate.

        Returns
        -------
        bool
            ``True`` when validation passes.

        Raises
        ------
        TypeError
            If *df* is not a :class:`pandas.DataFrame`.
        ValueError
            If *df* is empty, or contains negative rainfall/temperature
            values in the expected columns.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__!r}.")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        # Validate rainfall columns when present
        for col in ("min_rainfall_mm", "max_rainfall_mm"):
            if col in df.columns:
                if (df[col] < 0).any():
                    raise ValueError(
                        f"Column {col!r} contains negative values, which are physically impossible."
                    )

        # Validate min <= max for rainfall
        if "min_rainfall_mm" in df.columns and "max_rainfall_mm" in df.columns:
            if (df["min_rainfall_mm"] > df["max_rainfall_mm"]).any():
                raise ValueError(
                    "Some rows have min_rainfall_mm > max_rainfall_mm."
                )

        return True

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a cleaned copy of *df* with normalised column names.

        Performs the following transformations (all non-mutating):

        * Drops rows that are entirely ``NaN``.
        * Strips whitespace and lower-cases column names; replaces spaces
          with underscores.
        * Strips leading/trailing whitespace from string columns.
        * Converts boolean-like string values (``"true"``/``"false"``) in
          ``native``, ``drought_tolerant``, and ``suitable_for_agroforestry``
          columns to Python ``bool``.

        Parameters
        ----------
        df:
            Raw input DataFrame.

        Returns
        -------
        pd.DataFrame
            Cleaned copy; the original *df* is never modified.
        """
        cleaned = df.copy()
        cleaned = cleaned.dropna(how="all")
        cleaned.columns = [c.lower().strip().replace(" ", "_") for c in cleaned.columns]

        # Strip string whitespace
        str_cols = cleaned.select_dtypes(include="object").columns
        for col in str_cols:
            cleaned = cleaned.assign(**{col: cleaned[col].str.strip()})

        # Normalise boolean columns
        bool_cols = [c for c in ("native", "drought_tolerant", "suitable_for_agroforestry") if c in cleaned.columns]
        for col in bool_cols:
            if cleaned[col].dtype == object:
                cleaned = cleaned.assign(
                    **{col: cleaned[col].str.lower().map({"true": True, "false": False, "1": True, "0": False})}
                )

        # Normalise climate_zone to lowercase
        if "climate_zone" in cleaned.columns:
            cleaned = cleaned.assign(climate_zone=cleaned["climate_zone"].str.lower())

        # Normalise soil_type to lowercase
        if "soil_type" in cleaned.columns:
            cleaned = cleaned.assign(soil_type=cleaned["soil_type"].str.lower())

        return cleaned

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter(
        self,
        df: pd.DataFrame,
        *,
        climate_zone: Optional[str] = None,
        min_rainfall_mm: Optional[float] = None,
        max_rainfall_mm: Optional[float] = None,
        soil_type: Optional[str] = None,
        native_only: bool = False,
        drought_tolerant: Optional[bool] = None,
        suitable_for_agroforestry: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Filter species by environmental criteria.

        All criteria are applied as logical AND.  Passing no criteria returns
        the full (preprocessed) DataFrame.

        Parameters
        ----------
        df:
            Input species DataFrame (raw or preprocessed).
        climate_zone:
            One of ``"tropical"``, ``"subtropical"``, ``"temperate"``,
            ``"boreal"``, ``"arid"``.  Case-insensitive.
        min_rainfall_mm:
            Include only species whose *max_rainfall_mm* is at or above this
            value (i.e. the species can survive at least this much rain).
        max_rainfall_mm:
            Include only species whose *min_rainfall_mm* is at or below this
            value (i.e. the species can survive with at most this much rain).
        soil_type:
            Filter by soil type.  Case-insensitive.
        native_only:
            When ``True``, return only native species.
        drought_tolerant:
            When set, filter by drought tolerance flag.
        suitable_for_agroforestry:
            When set, filter by agroforestry suitability flag.

        Returns
        -------
        pd.DataFrame
            Filtered subset (new DataFrame; original is not modified).

        Raises
        ------
        ValueError
            If *climate_zone* or *soil_type* is not a recognised value.
        ValueError
            If *min_rainfall_mm* or *max_rainfall_mm* is negative.
        """
        # --- input validation ---
        if climate_zone is not None:
            zone_lower = climate_zone.lower()
            if zone_lower not in VALID_CLIMATE_ZONES:
                raise ValueError(
                    f"Invalid climate_zone {climate_zone!r}. "
                    f"Must be one of {sorted(VALID_CLIMATE_ZONES)}."
                )
        else:
            zone_lower = None

        if soil_type is not None:
            soil_lower = soil_type.lower()
            if soil_lower not in VALID_SOIL_TYPES:
                raise ValueError(
                    f"Invalid soil_type {soil_type!r}. "
                    f"Must be one of {sorted(VALID_SOIL_TYPES)}."
                )
        else:
            soil_lower = None

        if min_rainfall_mm is not None and min_rainfall_mm < 0:
            raise ValueError(f"min_rainfall_mm must be >= 0, got {min_rainfall_mm}.")
        if max_rainfall_mm is not None and max_rainfall_mm < 0:
            raise ValueError(f"max_rainfall_mm must be >= 0, got {max_rainfall_mm}.")

        # --- preprocess then filter (immutable: work on copies) ---
        result = self.preprocess(df)

        if zone_lower is not None and "climate_zone" in result.columns:
            result = result[result["climate_zone"] == zone_lower]

        if min_rainfall_mm is not None and "max_rainfall_mm" in result.columns:
            result = result[result["max_rainfall_mm"] >= min_rainfall_mm]

        if max_rainfall_mm is not None and "min_rainfall_mm" in result.columns:
            result = result[result["min_rainfall_mm"] <= max_rainfall_mm]

        if soil_lower is not None and "soil_type" in result.columns:
            result = result[result["soil_type"] == soil_lower]

        if native_only and "native" in result.columns:
            result = result[result["native"] == True]  # noqa: E712

        if drought_tolerant is not None and "drought_tolerant" in result.columns:
            result = result[result["drought_tolerant"] == drought_tolerant]

        if suitable_for_agroforestry is not None and "suitable_for_agroforestry" in result.columns:
            result = result[result["suitable_for_agroforestry"] == suitable_for_agroforestry]

        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Scoring & ranking
    # ------------------------------------------------------------------

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute a composite suitability score for each species row.

        The score is a weighted sum of normalised indicators:

        * Carbon sequestration (40 %)
        * Growth rate (30 %)
        * Native status (15 %)
        * Agroforestry suitability (10 %)
        * Drought tolerance (5 %)

        Custom weights can be supplied via ``config["score_weights"]``.

        Parameters
        ----------
        df:
            Pre-filtered species DataFrame.  Must contain at least one
            scoreable column; raises ``ValueError`` if *df* is empty.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with a new ``suitability_score`` column (0–100).

        Raises
        ------
        ValueError
            If *df* is empty.
        """
        if df.empty:
            raise ValueError("Cannot score an empty DataFrame.")

        weights = self.config.get(
            "score_weights",
            {
                "carbon": _WEIGHT_CARBON,
                "growth": _WEIGHT_GROWTH,
                "native": _WEIGHT_NATIVE,
                "agroforestry": _WEIGHT_AGROFORESTRY,
                "drought": _WEIGHT_DROUGHT,
            },
        )

        result = df.copy()

        def _normalise(series: pd.Series) -> pd.Series:
            """Min-max normalise a numeric series to [0, 1]."""
            mn, mx = series.min(), series.max()
            if mx == mn:
                return pd.Series(np.ones(len(series)), index=series.index)
            return (series - mn) / (mx - mn)

        def _bool_to_float(series: pd.Series) -> pd.Series:
            """Convert boolean column to 0.0/1.0 float series."""
            return series.fillna(False).astype(float)

        composite = pd.Series(np.zeros(len(result)), index=result.index)

        if "carbon_seq_tc_ha_yr" in result.columns:
            composite = composite + weights.get("carbon", _WEIGHT_CARBON) * _normalise(
                result["carbon_seq_tc_ha_yr"]
            )
        if "growth_rate_m_yr" in result.columns:
            composite = composite + weights.get("growth", _WEIGHT_GROWTH) * _normalise(
                result["growth_rate_m_yr"]
            )
        if "native" in result.columns:
            composite = composite + weights.get("native", _WEIGHT_NATIVE) * _bool_to_float(
                result["native"]
            )
        if "suitable_for_agroforestry" in result.columns:
            composite = composite + weights.get("agroforestry", _WEIGHT_AGROFORESTRY) * _bool_to_float(
                result["suitable_for_agroforestry"]
            )
        if "drought_tolerant" in result.columns:
            composite = composite + weights.get("drought", _WEIGHT_DROUGHT) * _bool_to_float(
                result["drought_tolerant"]
            )

        result = result.assign(suitability_score=(composite * 100).round(2))
        return result

    def rank(self, df: pd.DataFrame, top_n: Optional[int] = None) -> pd.DataFrame:
        """Score and rank species by suitability, highest first.

        Parameters
        ----------
        df:
            Species DataFrame to rank.
        top_n:
            If given, return only the top *top_n* species.

        Returns
        -------
        pd.DataFrame
            Ranked copy of *df* with a ``suitability_score`` column and a
            ``rank`` column (1 = best).

        Raises
        ------
        ValueError
            If *df* is empty.
        ValueError
            If *top_n* is not a positive integer.
        """
        if df.empty:
            raise ValueError("Cannot rank an empty DataFrame.")
        if top_n is not None and (not isinstance(top_n, int) or top_n < 1):
            raise ValueError(f"top_n must be a positive integer, got {top_n!r}.")

        scored = self.score(df)
        ranked = (
            scored
            .sort_values("suitability_score", ascending=False)
            .reset_index(drop=True)
        )
        ranked = ranked.assign(rank=range(1, len(ranked) + 1))

        if top_n is not None:
            ranked = ranked.head(top_n).reset_index(drop=True)

        return ranked

    # ------------------------------------------------------------------
    # Legacy / generic analysis pipeline
    # ------------------------------------------------------------------

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run a generic descriptive analysis and return summary metrics.

        This method is kept for backward compatibility with earlier versions
        of the tool.  For species-specific filtering and ranking use
        :meth:`filter` and :meth:`rank` instead.

        Parameters
        ----------
        df:
            Input species DataFrame (raw or preprocessed).

        Returns
        -------
        dict
            Dictionary with keys:

            * ``total_records`` – row count after preprocessing.
            * ``columns`` – list of column names.
            * ``missing_pct`` – per-column missing-value percentages.
            * ``summary_stats`` – descriptive stats for numeric columns
              (present only when numeric columns exist).
            * ``totals`` – column sums for numeric columns.
            * ``means`` – column means for numeric columns.

        Raises
        ------
        ValueError
            If *df* is empty.
        """
        self.validate(df)
        preprocessed = self.preprocess(df)
        result: Dict[str, Any] = {
            "total_records": len(preprocessed),
            "columns": list(preprocessed.columns),
            "missing_pct": (
                preprocessed.isnull().sum() / len(preprocessed) * 100
            ).round(1).to_dict(),
        }
        numeric_df = preprocessed.select_dtypes(include="number")
        if not numeric_df.empty:
            result["summary_stats"] = numeric_df.describe().round(3).to_dict()
            result["totals"] = numeric_df.sum().round(2).to_dict()
            result["means"] = numeric_df.mean().round(3).to_dict()
        return result

    def run(self, filepath: str) -> Dict[str, Any]:
        """Full pipeline: load -> validate -> analyze.

        Convenience wrapper that chains :meth:`load_data`, :meth:`validate`,
        and :meth:`analyze`.

        Parameters
        ----------
        filepath:
            Path to the CSV or Excel input file.

        Returns
        -------
        dict
            Analysis result dictionary (see :meth:`analyze`).
        """
        df = self.load_data(filepath)
        self.validate(df)
        return self.analyze(df)

    def to_dataframe(self, result: Dict[str, Any]) -> pd.DataFrame:
        """Convert a flat or nested analysis result dict to a DataFrame.

        Each scalar value becomes a row.  Nested dicts are flattened with
        dot-separated keys (e.g. ``"totals.carbon_seq_tc_ha_yr"``).

        Parameters
        ----------
        result:
            Dictionary returned by :meth:`analyze`.

        Returns
        -------
        pd.DataFrame
            Two-column DataFrame with ``metric`` and ``value`` columns.
        """
        rows: List[Dict[str, Any]] = []
        for k, v in result.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    rows.append({"metric": f"{k}.{kk}", "value": vv})
            else:
                rows.append({"metric": k, "value": v})
        return pd.DataFrame(rows)
