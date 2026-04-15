"""
Basic usage example for tree-species-selector.

Demonstrates the full pipeline: load -> filter -> score -> rank,
as well as the legacy analyze() method and the domain-aligned data generator.

Run from the project root:
    python examples/basic_usage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path when running from examples/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import SpeciesSelector
from src.data_generator import generate_domain_sample

DEMO_CSV = Path(__file__).parent.parent / "demo" / "sample_data.csv"


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main() -> None:
    """Run the full demonstration pipeline."""
    selector = SpeciesSelector()

    # ------------------------------------------------------------------
    # 1. Load from CSV
    # ------------------------------------------------------------------
    print_section("1. Load Dataset")
    df = selector.load_data(str(DEMO_CSV))
    print(f"Loaded {len(df)} species from {DEMO_CSV.name}")
    print(f"Columns: {list(df.columns)}")

    # ------------------------------------------------------------------
    # 2. Filter by environmental criteria
    # ------------------------------------------------------------------
    print_section("2. Filter: Tropical + Loam + Rainfall 1200-2500 mm")
    filtered = selector.filter(
        df,
        climate_zone="tropical",
        min_rainfall_mm=1200,
        max_rainfall_mm=2500,
        soil_type="loam",
    )
    print(f"Matching species: {len(filtered)}")
    print(filtered[["species_name", "climate_zone", "soil_type", "growth_rate_m_yr"]].to_string(index=False))

    # ------------------------------------------------------------------
    # 3. Rank top-5 tropical species
    # ------------------------------------------------------------------
    print_section("3. Rank Top 5 Tropical Species")
    tropical = selector.filter(df, climate_zone="tropical")
    ranked = selector.rank(tropical, top_n=5)
    print(
        ranked[["rank", "species_name", "suitability_score", "growth_rate_m_yr", "carbon_seq_tc_ha_yr"]]
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # 4. Full analysis pipeline
    # ------------------------------------------------------------------
    print_section("4. Full Analysis Pipeline (analyze)")
    analysis = selector.run(str(DEMO_CSV))
    print(f"Total records : {analysis['total_records']}")
    print(f"Columns       : {analysis['columns']}")
    if "means" in analysis:
        print("Column means:")
        for col, mean_val in list(analysis["means"].items())[:4]:
            print(f"  {col}: {mean_val:.3f}")

    # ------------------------------------------------------------------
    # 5. Domain-aligned data generator
    # ------------------------------------------------------------------
    print_section("5. Domain-Aligned Data Generator")
    generated = generate_domain_sample()
    print(f"Generated {len(generated)} species from built-in pool")
    print(generated[["species_name", "climate_zone", "growth_rate_m_yr"]].to_string(index=False))

    # ------------------------------------------------------------------
    # 6. Custom scoring weights
    # ------------------------------------------------------------------
    print_section("6. Custom Weights: Carbon-Focused Ranking")
    carbon_selector = SpeciesSelector(
        config={
            "score_weights": {
                "carbon": 0.70,
                "growth": 0.20,
                "native": 0.05,
                "agroforestry": 0.03,
                "drought": 0.02,
            }
        }
    )
    carbon_ranked = carbon_selector.rank(df, top_n=5)
    print("Top 5 by carbon-focused score:")
    print(
        carbon_ranked[["rank", "species_name", "suitability_score", "carbon_seq_tc_ha_yr"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
