# Changelog

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
