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


# Default category catalogue for binary spam classification.
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

# Fine-grained spam sub-categories for multiclass classification.
SPAM_SUBCATEGORIES: tuple[Category, ...] = (
    Category(
        name="phishing",
        label="Phishing",
        description=(
            "Attempts to steal credentials or personal"
            " information by impersonating trusted entities."
        ),
    ),
    Category(
        name="malware",
        label="Malware",
        description=(
            "Messages containing or linking to malicious"
            " software, viruses, or trojans."
        ),
    ),
    Category(
        name="financial_fraud",
        label="Financial Fraud",
        description=(
            "Fraudulent investment schemes, advance-fee"
            " fraud, or fake financial offers."
        ),
    ),
    Category(
        name="lottery",
        label="Lottery",
        description=(
            "Fake lottery or prize-winning notifications"
            " requesting fees or personal details."
        ),
    ),
    Category(
        name="romance_scam",
        label="Romance Scam",
        description=(
            "Romantic or emotional manipulation to extract"
            " money or personal information."
        ),
    ),
    Category(
        name="drug",
        label="Drug",
        description=(
            "Unsolicited advertisements for"
            " pharmaceuticals, supplements, or illegal drugs."
        ),
    ),
    Category(
        name="fake_service",
        label="Fake Service",
        description=(
            "Fake service offers such as SEO, web design,"
            " or business consulting spam."
        ),
    ),
    Category(
        name="tech_support",
        label="Tech Support",
        description=(
            "Fake tech support scams claiming device"
            " infection or account compromise."
        ),
    ),
)

# Combined catalogue: binary + multiclass categories.
MULTICLASS_CATEGORIES: tuple[Category, ...] = (
    Category(
        name="ham",
        label="Ham",
        description="Legitimate, non-spam message.",
    ),
    *SPAM_SUBCATEGORIES,
)


def get_category(name: str) -> Category:
    """Look up a category by *name* from binary or multiclass catalogues.

    Searches :data:`SPAM_CATEGORIES` first, then :data:`MULTICLASS_CATEGORIES`.

    Args:
        name: Category identifier.

    Returns:
        The matching :class:`Category`.

    Raises:
        KeyError: If no category with the given name exists.
    """
    for cat in (*SPAM_CATEGORIES, *SPAM_SUBCATEGORIES):
        if cat.name == name:
            return cat
    raise KeyError(f"Unknown category: {name!r}")
