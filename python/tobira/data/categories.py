"""Category definitions for synthetic data generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    """A category used for synthetic data generation.

    Attributes:
        name: Machine-readable identifier (e.g. ``spam``, ``ham``).
        label: Human-readable display name.
        description: Short description used in LLM prompts.
    """

    name: str
    label: str
    description: str


# Default category catalogue for spam classification.
SPAM_CATEGORIES: tuple[Category, ...] = (
    Category(
        name="spam",
        label="Spam",
        description="Unsolicited commercial or fraudulent message.",
    ),
    Category(
        name="ham",
        label="Ham",
        description="Legitimate, non-spam message.",
    ),
)


def get_category(name: str) -> Category:
    """Look up a default category by *name*.

    Args:
        name: Category identifier.

    Returns:
        The matching :class:`Category`.

    Raises:
        KeyError: If no category with the given name exists.
    """
    for cat in SPAM_CATEGORIES:
        if cat.name == name:
            return cat
    raise KeyError(f"Unknown category: {name!r}")
