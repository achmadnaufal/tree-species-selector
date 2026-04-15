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
"""

from src.main import VALID_CLIMATE_ZONES, VALID_SOIL_TYPES, SpeciesSelector
from src.data_generator import generate_domain_sample, generate_sample

__all__ = [
    "SpeciesSelector",
    "VALID_CLIMATE_ZONES",
    "VALID_SOIL_TYPES",
    "generate_sample",
    "generate_domain_sample",
]
