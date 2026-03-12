"""tobira.data - synthetic data generation for training and evaluation."""

from tobira.data.categories import SPAM_CATEGORIES, Category, get_category
from tobira.data.generator import SyntheticSample, generate

__all__ = [
    "Category",
    "SPAM_CATEGORIES",
    "SyntheticSample",
    "generate",
    "get_category",
]
