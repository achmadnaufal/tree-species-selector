![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-60%2B%20passing-brightgreen)
![Last Commit](https://img.shields.io/github/last-commit/achmadnaufal/tree-species-selector)

# Tree Species Selector

A decision-support library that filters, scores, and ranks tree species by
climate zone, soil type, rainfall tolerance, and ecological traits — helping
foresters and land-use planners choose the best candidates for reforestation
and agroforestry projects.

---

## Features

- **Multi-criteria filtering** — narrow candidates by climate zone, rainfall range, soil type, native status, drought tolerance, and agroforestry suitability
- **Composite suitability scoring** — weighted index combining carbon sequestration, growth rate, native status, agroforestry fit, and drought tolerance
- **Configurable weights** — override default scoring weights via a config dict
- **Immutable pipeline** — original DataFrames are never mutated; every operation returns a new copy
- **Input validation** — rejects negative rainfall, invalid climate zones, and malformed data with clear error messages
- **CSV and Excel support** — load `.csv`, `.xlsx`, or `.xls` files
- **30-species demo dataset** — realistic tropical, subtropical, temperate, and boreal species with Indonesian/tropical reforestation context
- **60+ pytest tests** — full coverage of filtering, scoring, ranking, edge cases, and the data generator

---

## Quick Start

```bash
git clone https://github.com/achmadnaufal/tree-species-selector.git
cd tree-species-selector
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

```python
from src.main import SpeciesSelector

selector = SpeciesSelector()
df = selector.load_data("demo/sample_data.csv")
ranked = selector.rank(df, top_n=5)
print(ranked[["rank", "species_name", "suitability_score"]])
```

---

## Usage

### 1. Load and inspect the dataset

```python
from src.main import SpeciesSelector

selector = SpeciesSelector()
df = selector.load_data("demo/sample_data.csv")
print(f"Loaded {len(df)} species")
```

### 2. Filter by environmental criteria

```python
filtered = selector.filter(
    df,
    climate_zone="tropical",
    min_rainfall_mm=1200,
    max_rainfall_mm=2500,
    soil_type="loam",
)
print(filtered[["species_name", "climate_zone", "soil_type", "growth_rate_m_yr"]])
```

### 3. Rank species by suitability score

```python
tropical = selector.filter(df, climate_zone="tropical")
ranked = selector.rank(tropical, top_n=5)
print(ranked[["rank", "species_name", "suitability_score", "growth_rate_m_yr", "carbon_seq_tc_ha_yr"]])
```

### 4. Customise scoring weights

```python
selector = SpeciesSelector(
    config={
        "score_weights": {
            "carbon":       0.50,   # prioritise carbon sequestration
            "growth":       0.30,
            "native":       0.10,
            "agroforestry": 0.05,
            "drought":      0.05,
        }
    }
)
ranked = selector.rank(df, top_n=5)
```

### 5. Full analysis pipeline

```python
result = selector.run("demo/sample_data.csv")
print(f"Total records: {result['total_records']}")
print("Column means:", result.get("means"))
```

---

## Sample Output

```
=== Load Dataset ===
Loaded 30 species from demo/sample_data.csv

=== Filter: Tropical + Loam Soil + Rainfall 1200-2500 mm ===
     species_name climate_zone soil_type  growth_rate_m_yr
             Teak     tropical      loam               1.5
      Rubber Tree     tropical      loam               1.0
           Sengon     tropical      loam               3.5
           Durian     tropical      loam               0.8
        Jackfruit     tropical      loam               1.1
         Albizzia     tropical      loam               2.0

=== Top 5 Tropical Species by Suitability ===
 rank         species_name  suitability_score  growth_rate_m_yr  carbon_seq_tc_ha_yr
    1  Bamboo (Petung)               100.0               4.0                 16.5
    2  Bamboo (Moso)                  97.8               3.0                 15.0
    3         Sengon                  85.3               3.5                 13.2
    4         Acacia                  74.1               2.5                 11.4
    5          Jabon                  68.9               2.8                 10.7
```

---

## Running Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=term-missing
```

Expected output:

```
tests/test_selector.py::TestFilterByClimateZone::test_tropical_returns_only_tropical_species PASSED
tests/test_selector.py::TestFilterByClimateZone::test_boreal_returns_only_boreal_species PASSED
...
tests/test_data_generator.py::TestGenerateSample::test_returns_dataframe PASSED
tests/test_data_generator.py::TestGenerateDomainSample::test_required_columns_present PASSED
...
============================== 60+ passed in 0.35s ==============================
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.9+** | Core language |
| **pandas** | Data loading, filtering, and transformation |
| **NumPy** | Numeric normalization for scoring |
| **pytest** | Unit and integration testing |
| **pytest-cov** | Test coverage reporting |
| **Rich** | (available) Terminal formatting |
| **openpyxl** | Excel file support |

---

## Architecture

```mermaid
flowchart LR
    A["CSV / Excel File"] -->|load_data| B["Raw DataFrame"]
    B -->|validate| C{"Validation\n(schema, ranges)"}
    C -->|fail| X["ValueError / TypeError"]
    C -->|pass| D["preprocess\n(clean, normalize)"]
    D --> E["filter\n(climate, soil,\nrainfall, flags)"]
    E --> F["score\n(weighted composite)"]
    F --> G["rank\n(sort + top-N)"]
    G --> H["Ranked DataFrame\nwith suitability_score"]

    style A fill:#e0f0ff,stroke:#369
    style H fill:#d4edda,stroke:#28a745
    style X fill:#f8d7da,stroke:#dc3545
```

### Data flow

1. **Load** — `load_data()` reads CSV or Excel into a raw pandas DataFrame.
2. **Validate** — `validate()` checks for empty data, negative rainfall, and min/max consistency.
3. **Preprocess** — `preprocess()` normalizes column names, strips whitespace, converts boolean strings, and lowercases categorical fields. Returns a new DataFrame (immutable).
4. **Filter** — `filter()` applies AND-combined environmental criteria (climate zone, rainfall range, soil type, native, drought-tolerant, agroforestry). Returns a new subset.
5. **Score** — `score()` computes a 0-100 composite suitability score using configurable weights across carbon sequestration (40%), growth rate (30%), native status (15%), agroforestry suitability (10%), and drought tolerance (5%).
6. **Rank** — `rank()` sorts by score descending, adds a rank column, and optionally truncates to `top_n`.

---

## Project Structure

```
tree-species-selector/
├── src/
│   ├── __init__.py              # Public API exports
│   ├── main.py                  # SpeciesSelector core class
│   └── data_generator.py        # Sample data generators (legacy + domain-aligned)
├── demo/
│   └── sample_data.csv          # 30-row realistic species dataset
├── sample_data/
│   └── sample_data.csv          # Lightweight sample for quick testing
├── tests/
│   ├── __init__.py
│   ├── test_selector.py         # 50+ assertions for SpeciesSelector
│   └── test_data_generator.py   # 15+ assertions for data generator
├── examples/
│   └── basic_usage.py           # Runnable usage example
├── data/                        # Drop your own data files here (gitignored)
├── .gitignore
├── CHANGELOG.md
├── LICENSE
├── requirements.txt
└── README.md
```

---

> Built by [Achmad Naufal](https://github.com/achmadnaufal) | Lead Data Analyst | Power BI · SQL · Python · GIS
