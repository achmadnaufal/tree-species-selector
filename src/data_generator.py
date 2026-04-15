"""
Generate realistic tree-species sample data for testing and demonstration.

Run this script directly to produce a CSV file:

    python src/data_generator.py

The generated data follows the schema expected by :class:`src.main.SpeciesSelector`
and is useful for exploratory testing and CI smoke tests.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Column definitions for the generic (legacy) generator
_LEGACY_COLUMNS: tuple[str, ...] = (
    "zone_id",
    "climate_type",
    "rainfall_mm",
    "soil_pH",
    "altitude_m",
    "species",
    "suitability_score",
)

# ---------------------------------------------------------------------------
# Realistic species pool used by the domain-aware generator
# ---------------------------------------------------------------------------
_SPECIES_POOL: tuple[dict, ...] = (
    {
        "species_name": "Teak",
        "scientific_name": "Tectona grandis",
        "climate_zone": "tropical",
        "min_rainfall_mm": 1200,
        "max_rainfall_mm": 2500,
        "min_temp_c": 20,
        "max_temp_c": 35,
        "soil_type": "loam",
        "growth_rate_m_yr": 1.5,
        "carbon_seq_tc_ha_yr": 8.2,
        "native": True,
        "drought_tolerant": True,
        "suitable_for_agroforestry": True,
    },
    {
        "species_name": "Acacia",
        "scientific_name": "Acacia mangium",
        "climate_zone": "tropical",
        "min_rainfall_mm": 1000,
        "max_rainfall_mm": 3500,
        "min_temp_c": 18,
        "max_temp_c": 38,
        "soil_type": "sandy_loam",
        "growth_rate_m_yr": 2.5,
        "carbon_seq_tc_ha_yr": 11.4,
        "native": True,
        "drought_tolerant": True,
        "suitable_for_agroforestry": True,
    },
    {
        "species_name": "Eucalyptus",
        "scientific_name": "Eucalyptus pellita",
        "climate_zone": "tropical",
        "min_rainfall_mm": 1000,
        "max_rainfall_mm": 2500,
        "min_temp_c": 15,
        "max_temp_c": 40,
        "soil_type": "sandy",
        "growth_rate_m_yr": 3.0,
        "carbon_seq_tc_ha_yr": 12.0,
        "native": False,
        "drought_tolerant": True,
        "suitable_for_agroforestry": False,
    },
    {
        "species_name": "Sengon",
        "scientific_name": "Paraserianthes falcataria",
        "climate_zone": "tropical",
        "min_rainfall_mm": 1200,
        "max_rainfall_mm": 3000,
        "min_temp_c": 18,
        "max_temp_c": 36,
        "soil_type": "loam",
        "growth_rate_m_yr": 3.5,
        "carbon_seq_tc_ha_yr": 13.2,
        "native": True,
        "drought_tolerant": False,
        "suitable_for_agroforestry": True,
    },
    {
        "species_name": "Jabon",
        "scientific_name": "Neolamarckia cadamba",
        "climate_zone": "tropical",
        "min_rainfall_mm": 1500,
        "max_rainfall_mm": 4000,
        "min_temp_c": 22,
        "max_temp_c": 38,
        "soil_type": "clay",
        "growth_rate_m_yr": 2.8,
        "carbon_seq_tc_ha_yr": 10.7,
        "native": True,
        "drought_tolerant": False,
        "suitable_for_agroforestry": True,
    },
    {
        "species_name": "Neem",
        "scientific_name": "Azadirachta indica",
        "climate_zone": "tropical",
        "min_rainfall_mm": 400,
        "max_rainfall_mm": 1000,
        "min_temp_c": 20,
        "max_temp_c": 45,
        "soil_type": "sandy_loam",
        "growth_rate_m_yr": 1.3,
        "carbon_seq_tc_ha_yr": 4.5,
        "native": True,
        "drought_tolerant": True,
        "suitable_for_agroforestry": True,
    },
    {
        "species_name": "Oak",
        "scientific_name": "Quercus robur",
        "climate_zone": "temperate",
        "min_rainfall_mm": 500,
        "max_rainfall_mm": 1200,
        "min_temp_c": 5,
        "max_temp_c": 25,
        "soil_type": "loam",
        "growth_rate_m_yr": 0.6,
        "carbon_seq_tc_ha_yr": 4.3,
        "native": True,
        "drought_tolerant": False,
        "suitable_for_agroforestry": False,
    },
    {
        "species_name": "Scots Pine",
        "scientific_name": "Pinus sylvestris",
        "climate_zone": "boreal",
        "min_rainfall_mm": 300,
        "max_rainfall_mm": 800,
        "min_temp_c": -5,
        "max_temp_c": 25,
        "soil_type": "sandy",
        "growth_rate_m_yr": 0.9,
        "carbon_seq_tc_ha_yr": 4.8,
        "native": True,
        "drought_tolerant": True,
        "suitable_for_agroforestry": False,
    },
)


def generate_sample(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic species-like dataset for legacy/generic testing.

    The output schema uses simplified column names (``zone_id``, ``climate_type``,
    etc.) and is intended for unit/smoke tests that do not require the full
    domain schema expected by :class:`~src.main.SpeciesSelector`.

    Parameters
    ----------
    n:
        Number of rows to generate.  Must be a positive integer.
    seed:
        Random seed for reproducibility.  Defaults to ``42``.

    Returns
    -------
    pd.DataFrame
        Newly created DataFrame with :data:`_LEGACY_COLUMNS` columns and
        *n* rows.  The original seed state is preserved after the call
        (a copy of the global random state is used internally).

    Raises
    ------
    ValueError
        If *n* is not a positive integer.

    Examples
    --------
    >>> df = generate_sample(100, seed=0)
    >>> len(df)
    100
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}.")

    rng = np.random.default_rng(seed)
    local_random = random.Random(seed)

    base_date = datetime(2023, 1, 1)
    n_groups = max(5, n // 20)

    data: dict[str, list] = {}
    for col in _LEGACY_COLUMNS:
        if "date" in col:
            data[col] = [
                (base_date + timedelta(days=local_random.randint(0, 365))).strftime("%Y-%m-%d")
                for _ in range(n)
            ]
        elif "id" in col or "code" in col:
            data[col] = [f"{col[:3].upper()}{local_random.randint(1, n_groups):03d}" for _ in range(n)]
        elif "category" in col or "type" in col or "status" in col:
            data[col] = [local_random.choice(["A", "B", "C", "D"]) for _ in range(n)]
        elif "pct" in col or "rate" in col or "ratio" in col:
            data[col] = list(rng.uniform(0, 100, n).round(2))
        else:
            base = rng.exponential(100, n)
            noise = rng.normal(0, 10, n)
            data[col] = list(np.abs(base + noise).round(2))

    return pd.DataFrame(data)


def generate_domain_sample(
    species_pool: Optional[tuple[dict, ...]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a clean domain-aligned DataFrame drawn from a curated species pool.

    Unlike :func:`generate_sample`, this function produces records that match the
    full schema consumed by :class:`~src.main.SpeciesSelector` (``climate_zone``,
    ``soil_type``, ``carbon_seq_tc_ha_yr``, etc.).

    Parameters
    ----------
    species_pool:
        Sequence of species dictionaries to use as the source.  Defaults to
        the built-in :data:`_SPECIES_POOL`.
    seed:
        Random seed, currently unused but reserved for future stochastic
        extensions.

    Returns
    -------
    pd.DataFrame
        New immutable DataFrame with one row per species in *species_pool*.

    Examples
    --------
    >>> df = generate_domain_sample()
    >>> "climate_zone" in df.columns
    True
    """
    pool = species_pool if species_pool is not None else _SPECIES_POOL
    # Build a new list of dicts (never mutate the pool entries)
    rows = [dict(entry) for entry in pool]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    df = generate_sample(300)
    out_path = "data/sample.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} records -> {out_path}")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
