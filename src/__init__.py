"""
tree-species-selector
=====================

A decision-support library for filtering, scoring, and ranking tree species
by climate zone, soil type, rainfall tolerance, and ecological traits.

Public API
----------
- :class:`~src.main.SpeciesSelector` -- main entry point for all operations.
- :data:`~src.main.VALID_CLIMATE_ZONES` -- frozenset of recognised climate zones.
- :data:`~src.main.VALID_SOIL_TYPES` -- frozenset of recognised soil types.
- :func:`~src.data_generator.generate_sample` -- generate synthetic test data.
- :func:`~src.data_generator.generate_domain_sample` -- generate domain-aligned
  sample data matching the SpeciesSelector schema.
- :class:`~src.site_match_scorer.Site` -- describe a planting site.
- :func:`~src.site_match_scorer.score_site_match` -- score every species
  against a given site (rainfall, temperature, soil).
- :func:`~src.site_match_scorer.recommend_for_site` -- top-N species ranked
  by site match.
- :func:`~src.portfolio_builder.build_portfolio` -- build a diverse
  site-matched species portfolio with assigned proportions.
- :func:`~src.portfolio_builder.compare_portfolios` -- sensitivity analysis
  across fit-vs-diversity weights.
- :class:`~src.portfolio_builder.PortfolioResult` -- immutable result
  container returned by :func:`build_portfolio`.
"""

from src.main import VALID_CLIMATE_ZONES, VALID_SOIL_TYPES, SpeciesSelector
from src.data_generator import generate_domain_sample, generate_sample
from src.site_match_scorer import Site, recommend_for_site, score_site_match
from src.portfolio_builder import PortfolioResult, build_portfolio, compare_portfolios

__all__ = [
    "SpeciesSelector",
    "VALID_CLIMATE_ZONES",
    "VALID_SOIL_TYPES",
    "generate_sample",
    "generate_domain_sample",
    "Site",
    "score_site_match",
    "recommend_for_site",
    "PortfolioResult",
    "build_portfolio",
    "compare_portfolios",
]
