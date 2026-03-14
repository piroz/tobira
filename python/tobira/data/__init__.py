"""tobira.data - synthetic data generation for training and evaluation."""

from tobira.data.categories import (
    MULTICLASS_CATEGORIES,
    SPAM_CATEGORIES,
    SPAM_SUBCATEGORIES,
    Category,
    get_category,
)
from tobira.data.generator import SyntheticSample, generate

__all__ = [
    "Category",
    "MULTICLASS_CATEGORIES",
    "SPAM_CATEGORIES",
    "SPAM_SUBCATEGORIES",
    "SyntheticSample",
    "generate",
    "get_category",
]
