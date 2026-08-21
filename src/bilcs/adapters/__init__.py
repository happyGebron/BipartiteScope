"""Domain adapters that translate records into the canonical core schema."""

from .academic import AcademicExplorerAdapter
from .recommendation import RecommendationAdapter

__all__ = ["AcademicExplorerAdapter", "RecommendationAdapter"]
