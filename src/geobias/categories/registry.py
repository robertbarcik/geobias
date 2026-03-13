"""Registry of all test categories."""

from .base import TestCategory
from .narrative_framing import NarrativeFraming
from .refusal_asymmetry import RefusalAsymmetry
from .entity_sentiment import EntitySentiment
from .language_quality import LanguageQuality
from .regulatory_values import RegulatoryValues
from .self_awareness import SelfAwareness
from .historical_narrative import HistoricalNarrative

CATEGORY_REGISTRY: dict[str, TestCategory] = {
    "narrative_framing": NarrativeFraming(),
    "refusal_asymmetry": RefusalAsymmetry(),
    "entity_sentiment": EntitySentiment(),
    "language_quality": LanguageQuality(),
    "regulatory_values": RegulatoryValues(),
    "self_awareness": SelfAwareness(),
    "historical_narrative": HistoricalNarrative(),
}


def get_category(name: str) -> TestCategory:
    if name not in CATEGORY_REGISTRY:
        raise ValueError(f"Unknown category: {name}. Available: {list(CATEGORY_REGISTRY.keys())}")
    return CATEGORY_REGISTRY[name]


def get_categories(names: list[str]) -> list[TestCategory]:
    return [get_category(n) for n in names]
