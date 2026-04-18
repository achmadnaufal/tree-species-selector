# Changelog

All notable changes to this project are documented here.  This project
adheres to [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-04-19
### Added
- `src/portfolio_builder.py` -- new module that composes an optimal
  site-matched, diversity-aware species portfolio:
  - **`build_portfolio()`** -- greedy selector that seeds on the best
    site-match species and iteratively adds candidates maximising
    `alpha * site_fit + (1 - alpha) * functional_distance`, assigning
    proportions proportionally to the final objective.
  - **`compare_portfolios()`** -- sensitivity helper that runs
    `build_portfolio` across several `alpha` values and returns a tidy
    comparison DataFrame.
  - **`PortfolioResult`** -- immutable dataclass carrying the portfolio,
    site, alpha, weighted mean site-fit, and a human-readable summary.
  - Proportions always sum to 1.0; empty-pool and zero-objective cases
    fall back gracefully (empty portfolio with informative summary,
    equal-split proportions, respectively).
  - Configurable `portfolio_size`, `alpha`, `min_site_score`,
    `trait_columns`, and pass-through `site_match_kwargs`.
  - Comprehensive input validation (non-DataFrame, empty, non-positive
    sizes, out-of-range alpha / min_site_score) with clear messages.
  - Fully immutable: input DataFrame is never mutated, every function
    returns a new object.
- `tests/test_portfolio_builder.py` -- 38 pytest assertions covering
  happy paths, alpha-sweep behaviour (alpha=1 pure ranking, alpha=0
  diversity prevalence), input validation, no-match / single-candidate
  edge cases, missing-trait fallback, immutability, determinism, and
  the `compare_portfolios` helper.
- `README.md` restructured with the requested sections (Overview,
  Installation, Quick Start, Step-by-Step Usage, Selection Methodology,
  Species Database, License) and a new "Build a diverse species
  portfolio" step.
- `src/__init__.py` exports updated to include `build_portfolio`,
  `compare_portfolios`, and `PortfolioResult`.

## [0.3.0] - 2026-04-18
### Added
- `src/site_match_scorer.py` — new module that evaluates how well each
  candidate species fits a *given site* (rainfall, temperature, soil), as a
  complement to the trait-based suitability score in `SpeciesSelector`:
  - **`Site` dataclass** — immutable description of a planting location with
    input validation (negative rainfall and unknown soil types are rejected
    on construction).
  - **`score_site_match()`** — returns a copy of the species DataFrame with
    `rainfall_match`, `temperature_match`, `soil_match`, and the composite
    `site_match_score` columns, each in `[0, 1]`.
  - **`recommend_for_site()`** — convenience top-N ranker with a
    `min_score` threshold and added `rank` column.
  - Configurable rainfall/temperature tolerance margins, partial soil-match
    score, and sub-score weights.
  - Soil-compatibility groups (e.g. loam ↔ clay_loam ↔ sandy_loam) for
    partial soil matches.
  - Comprehensive input validation (empty DataFrame, missing required
    columns, non-positive tolerances, out-of-range partial score, negative
    or all-zero weights) with clear error messages.
  - Fully immutable: every function returns new objects; no input is mutated.
- `tests/test_site_match_scorer.py` — 30+ pytest assertions covering the
  `Site` dataclass, happy paths, envelope decay (linear and zero-clamped),
  boundary values, exact / compatible / unrelated soil matches, custom
  weights, input validation errors, immutability, determinism, and the
  `recommend_for_site` wrapper (top-N ordering, rank column, min-score
  filter, invalid arguments).
- README section "New: Site-Match Scorer" with a runnable Quick Start and
  step-by-step usage (describe site, score, customise tolerances/weights,
  ranked recommendation).
- `src/__init__.py` exports updated to include `Site`, `score_site_match`,
  and `recommend_for_site`.

## [Unreleased] - 2026-04-17
### Added
- `src/species_diversity_scorer.py` — new module that computes ecological and
  functional diversity indices for a proposed planting plan:
  - **Shannon entropy H'** (nats) — quantifies species richness weighted by
    abundance.
  - **Simpson diversity D** (0–1) — probability that two randomly drawn
    individuals belong to different species.
  - **Pielou's evenness J'** (0–1) — how uniformly proportions are
    distributed across species.
  - **Functional diversity** — mean pairwise Euclidean distance between
    species in normalised trait space (growth rate, carbon sequestration,
    rainfall tolerance, temperature range).
  - `compute_diversity()` — returns an immutable `DiversityResult` dataclass.
  - `score_plan_diversity()` — convenience wrapper returning a one-row pandas
    DataFrame for pipeline integration.
  - Full input validation (empty DataFrame, missing column, negative
    proportions, all-zero proportions) with clear error messages.
  - Immutable design: all functions return new objects; no input is mutated.
- `tests/test_species_diversity_scorer.py` — 31 pytest assertions covering
  happy paths, edge cases (single species, missing traits, zero proportions,
  custom column names), determinism, parametrized Shannon known-values, and
  `score_plan_diversity` integration.
- README section "New: Species Diversity Scorer" with five step-by-step usage
  examples (basic, index inspection, tidy DataFrame, scenario comparison,
  custom column and traits).

## [0.2.0] - 2026-04-16
### Added
- Unit tests with pytest covering SpeciesSelector (filter, score, rank, validate,
  preprocess, analyze, run, to_dataframe) and data_generator (generate_sample,
  generate_domain_sample) — 60+ assertions across two test modules
- `tests/test_data_generator.py` with full coverage of generate_sample and
  generate_domain_sample including edge cases and immutability checks
- Expanded `demo/sample_data.csv` to 30 realistic tropical, subtropical,
  temperate, and boreal species for Indonesian/tropical reforestation contexts
- Comprehensive docstrings and type hints throughout `src/data_generator.py`
- `generate_domain_sample()` function producing domain-aligned DataFrames that
  match the SpeciesSelector schema directly
- Input validation in `generate_sample()` (rejects non-positive and non-integer n)
- Immutable random state management in `generate_sample()` using
  `numpy.random.default_rng` and `random.Random` (no global state mutation)
- Improved `src/__init__.py` with full public API documentation and `__all__` exports

### Changed
- `src/data_generator.py` refactored: column definitions moved to module-level
  constants, boolean flags replaced with typed tuples, no mutable global state
- `CHANGELOG.md` date corrected to 2026-04-16

## [0.1.0] - 2024-01-01
### Added
- Initial project scaffold
- Core SpeciesSelector class with load, validate, preprocess, filter, score,
  rank, analyze, run, and to_dataframe pipeline
- Sample data generator
- Basic README and requirements
