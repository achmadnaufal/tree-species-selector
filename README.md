# Tree Species Selector

Decision support tool for optimal tree species selection by climate and soil zone.

## Features

- Load species data from CSV or Excel files
- Filter candidates by climate zone, rainfall range, soil type, and ecological flags
- Score and rank species by a configurable composite suitability index
- Input validation with clear, actionable error messages
- Immutable data pipeline — original DataFrames are never modified
- Sample dataset covering tropical, temperate, and boreal species
- Full pytest test suite

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Step 1 — Clone and install dependencies

```bash
git clone https://github.com/achmadnaufal/tree-species-selector.git
cd tree-species-selector
pip install -r requirements.txt
```

### Step 2 — Load the sample dataset

```python
from src.main import SpeciesSelector

selector = SpeciesSelector()
df = selector.load_data("demo/sample_data.csv")
print(df.head())
```

### Step 3 — Filter by environmental criteria

```python
# Tropical species that tolerate 1 200–2 500 mm annual rainfall on loam soil
filtered = selector.filter(
    df,
    climate_zone="tropical",
    min_rainfall_mm=1200,
    max_rainfall_mm=2500,
    soil_type="loam",
)
print(filtered[["species_name", "climate_zone", "soil_type"]])
```

### Step 4 — Rank by suitability score

```python
ranked = selector.rank(filtered, top_n=5)
print(ranked[["rank", "species_name", "suitability_score"]])
```

### Step 5 — Run the full pipeline from a file

```python
result = selector.run("demo/sample_data.csv")
print(f"Total records analysed: {result['total_records']}")
```

## Sample Data

The file `demo/sample_data.csv` contains 20 real-world tree species with the
following columns:

| Column | Description | Example |
|---|---|---|
| `species_name` | Common name | `Teak` |
| `scientific_name` | Binomial name | `Tectona grandis` |
| `climate_zone` | `tropical` / `subtropical` / `temperate` / `boreal` | `tropical` |
| `min_rainfall_mm` | Minimum annual rainfall (mm) | `1200` |
| `max_rainfall_mm` | Maximum annual rainfall (mm) | `2500` |
| `min_temp_c` | Minimum temperature (°C) | `20` |
| `max_temp_c` | Maximum temperature (°C) | `35` |
| `soil_type` | Preferred soil type | `loam` |
| `growth_rate_m_yr` | Average height growth (m/year) | `1.5` |
| `carbon_seq_tC_ha_yr` | Carbon sequestration (tC/ha/year) | `8.2` |
| `native` | Native to its natural range | `true` |
| `drought_tolerant` | Tolerates prolonged dry periods | `true` |
| `suitable_for_agroforestry` | Suitable for mixed farming systems | `true` |

Valid climate zones: `tropical`, `subtropical`, `temperate`, `boreal`, `arid`.

Valid soil types: `loam`, `clay`, `sandy`, `clay_loam`, `sandy_loam`, `silt`, `peat`.

## Example Code Snippets

### Filter drought-tolerant agroforestry species

```python
result = selector.filter(
    df,
    drought_tolerant=True,
    suitable_for_agroforestry=True,
)
print(result["species_name"].tolist())
```

### Use custom scoring weights

```python
selector_carbon = SpeciesSelector(
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
ranked = selector_carbon.rank(df)
```

### Export results to CSV

```python
ranked = selector.rank(df, top_n=10)
ranked.to_csv("results/top10_species.csv", index=False)
```

### Convert analysis summary to DataFrame

```python
result = selector.analyze(df)
summary_df = selector.to_dataframe(result)
print(summary_df)
```

## Running Tests

```bash
# Install test dependency
pip install pytest

# Run all tests with verbose output
pytest tests/ -v

# Run with coverage report
pip install pytest-cov
pytest tests/ --cov=src --cov-report=term-missing
```

Expected output:

```
tests/test_selector.py::TestFilterByClimateZone::test_tropical_returns_only_tropical_species PASSED
tests/test_selector.py::TestFilterByClimateZone::test_boreal_returns_only_boreal_species PASSED
...
============= 30 passed in 0.42s =============
```

## Project Structure

```
tree-species-selector/
├── src/
│   ├── __init__.py
│   ├── main.py            # SpeciesSelector — core filtering, scoring, analysis
│   └── data_generator.py  # Programmatic sample data generator
├── demo/
│   └── sample_data.csv    # 20-row realistic species dataset
├── tests/
│   ├── __init__.py
│   └── test_selector.py   # pytest unit tests (30+ assertions)
├── examples/
│   └── basic_usage.py     # Runnable usage example
├── data/                  # Drop your own data files here (gitignored)
├── CHANGELOG.md
├── requirements.txt
└── README.md
```

## License

MIT License — free to use, modify, and distribute.
