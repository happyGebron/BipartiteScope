"""BipartiteScope: reusable attributed bipartite community analysis."""

from .config import BuildConfig, QueryConfig
from .domain import CanonicalBipartiteGraph

__all__ = ["BuildConfig", "CanonicalBipartiteGraph", "QueryConfig"]
__version__ = "0.2.0"
