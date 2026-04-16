# Changelog

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
