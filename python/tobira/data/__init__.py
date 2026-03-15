"""tobira.data - synthetic data generation for training and evaluation."""

from tobira.data.categories import (
    LANGUAGE_CATEGORIES,
    MULTICLASS_CATEGORIES,
    SPAM_CATEGORIES,
    SPAM_SUBCATEGORIES,
    Category,
    get_categories_for_language,
    get_category,
)
from tobira.data.generator import SyntheticSample, generate

__all__ = [
    "Category",
    "LANGUAGE_CATEGORIES",
    "MULTICLASS_CATEGORIES",
    "SPAM_CATEGORIES",
    "SPAM_SUBCATEGORIES",
    "SyntheticSample",
    "generate",
    "get_categories_for_language",
    "get_category",
]
