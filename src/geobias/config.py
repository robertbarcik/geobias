"""Model configuration and grouping for GeoBias evaluations."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    """A model available for evaluation."""

    id: str  # OpenRouter model ID, e.g. "qwen/qwen3-8b"
    name: str  # Human-readable name
    origin: str  # Country/region: "CN", "US", "EU"
    params_b: float  # Approximate parameter count in billions
    family: str  # Model family, e.g. "Qwen", "Llama", "Mistral"

    def safe_filename(self) -> str:
        return self.id.replace("/", "--")


@dataclass
class ModelGroup:
    """A set of comparable models from different origins."""

    name: str  # e.g. "7b"
    description: str
    models: list[ModelSpec] = field(default_factory=list)


# Default model groups — update IDs after verifying OpenRouter availability
# Model IDs verified against OpenRouter API as of 2026-03-13
DEFAULT_MODEL_GROUPS: dict[str, ModelGroup] = {
    "3b": ModelGroup(
        name="3b",
        description="~3-4B parameter class models",
        models=[
            ModelSpec("qwen/qwen3-4b:free", "Qwen3 4B", "CN", 4.0, "Qwen"),
            ModelSpec("meta-llama/llama-3.2-3b-instruct", "Llama 3.2 3B", "US", 3.0, "Llama"),
            ModelSpec("google/gemma-3-4b-it", "Gemma 3 4B", "US", 4.0, "Gemma"),
            ModelSpec("mistralai/ministral-3b-2512", "Ministral 3B", "EU", 3.0, "Mistral"),
        ],
    ),
    "7b": ModelGroup(
        name="7b",
        description="~7-14B parameter class models",
        models=[
            ModelSpec("qwen/qwen3-8b", "Qwen3 8B", "CN", 8.0, "Qwen"),
            ModelSpec("qwen/qwen3.5-9b", "Qwen3.5 9B", "CN", 9.0, "Qwen"),
            ModelSpec("meta-llama/llama-3.1-8b-instruct", "Llama 3.1 8B", "US", 8.0, "Llama"),
            ModelSpec("microsoft/phi-4", "Phi-4", "US", 14.0, "Phi"),
            ModelSpec("mistralai/ministral-8b-2512", "Ministral 8B", "EU", 8.0, "Mistral"),
        ],
    ),
}

# Test categories
ALL_CATEGORIES = [
    "narrative_framing",
    "refusal_asymmetry",
    "entity_sentiment",
    "language_quality",
    "regulatory_values",
    "self_awareness",
    "historical_narrative",
]


# Default evaluator panel — one from each major origin for cross-validation.
# Each evaluator judges every response independently. Where they agree, confidence
# is high; where they disagree, that disagreement is itself informative data.
DEFAULT_EVALUATORS: list[ModelSpec] = [
    ModelSpec("qwen/qwen3-235b-a22b-2507", "Qwen3 235B", "CN", 235.0, "Qwen"),
    ModelSpec("meta-llama/llama-3.3-70b-instruct", "Llama 3.3 70B", "US", 70.0, "Llama"),
    ModelSpec("mistralai/mistral-large-2512", "Mistral Large 3", "EU", 123.0, "Mistral"),
]

# Single cheap evaluator for budget runs (still multi-origin recommended)
BUDGET_EVALUATOR = ModelSpec("qwen/qwen3-235b-a22b-2507", "Qwen3 235B", "CN", 235.0, "Qwen")


@dataclass
class EvalConfig:
    """Full configuration for an evaluation run."""

    models: list[ModelSpec]
    categories: list[str]
    cases_per_category: int = 8
    evaluators: list[ModelSpec] = field(default_factory=lambda: list(DEFAULT_EVALUATORS))
    output_dir: Path = Path("reports")
    group_name: Optional[str] = None
    dry_run: bool = False
    resume_dir: Optional[Path] = None

    def __post_init__(self):
        for cat in self.categories:
            if cat not in ALL_CATEGORIES:
                raise ValueError(f"Unknown category: {cat}. Valid: {ALL_CATEGORIES}")

    @property
    def orchestrator_model(self) -> str:
        """Backward compat — returns first evaluator for test case generation."""
        return self.evaluators[0].id

    def run_dir(self) -> Path:
        """Create and return the directory for this run."""
        if self.resume_dir:
            self.resume_dir.mkdir(parents=True, exist_ok=True)
            return self.resume_dir

        from datetime import datetime

        label = self.group_name or "custom"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        d = self.output_dir / f"{label}-run-{stamp}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def to_dict(self) -> dict:
        return {
            "models": [asdict(m) for m in self.models],
            "categories": self.categories,
            "cases_per_category": self.cases_per_category,
            "evaluators": [asdict(e) for e in self.evaluators],
            "group_name": self.group_name,
        }


def _unknown_evaluator(model_id: str) -> ModelSpec:
    """Build a spec for a custom evaluator that is not in the known panel."""
    logger.warning(
        f"Evaluator '{model_id}' is not in the known panel, so its origin is 'unknown'. "
        f"The own-origin leniency analysis will be SKIPPED for this evaluator "
        f"(there is no origin to compare against). Its scores still count toward consensus."
    )
    return ModelSpec(model_id, model_id.split("/")[-1], "unknown", 0, "unknown")


def get_evaluators() -> list[ModelSpec]:
    """Get evaluator models from env or defaults."""
    raw = os.environ.get("GEOBIAS_EVALUATORS")
    if raw:
        ids = [s.strip() for s in raw.split(",") if s.strip()]
        all_known = {m.id: m for m in DEFAULT_EVALUATORS}
        return [all_known[mid] if mid in all_known else _unknown_evaluator(mid) for mid in ids]
    return list(DEFAULT_EVALUATORS)


def resolve_evaluators(evaluator_ids: Optional[list[str]] = None) -> list[ModelSpec]:
    """Resolve evaluator specs from explicit IDs, or fall back to env/defaults."""
    if not evaluator_ids:
        return get_evaluators()
    all_known = {m.id: m for m in DEFAULT_EVALUATORS}
    return [all_known[eid] if eid in all_known else _unknown_evaluator(eid) for eid in evaluator_ids]


def get_cases_per_category() -> int:
    raw = os.environ.get("GEOBIAS_CASES_PER_CATEGORY", "8")
    return int(raw)


def resolve_models(
    group: Optional[str] = None,
    model_ids: Optional[list[str]] = None,
) -> list[ModelSpec]:
    """Resolve model specs from group name or explicit IDs."""
    if model_ids:
        specs = []
        all_known = {m.id: m for g in DEFAULT_MODEL_GROUPS.values() for m in g.models}
        for mid in model_ids:
            if mid in all_known:
                specs.append(all_known[mid])
            else:
                # Unknown model — create minimal spec
                specs.append(ModelSpec(mid, mid.split("/")[-1], "unknown", 0, "unknown"))
        return specs

    if group == "all":
        seen = set()
        specs = []
        for g in DEFAULT_MODEL_GROUPS.values():
            for m in g.models:
                if m.id not in seen:
                    seen.add(m.id)
                    specs.append(m)
        return specs

    if group in DEFAULT_MODEL_GROUPS:
        return DEFAULT_MODEL_GROUPS[group].models

    raise ValueError(f"Unknown group: {group}. Available: {list(DEFAULT_MODEL_GROUPS.keys())} or 'all'")
