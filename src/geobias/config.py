"""Model configuration and grouping for GeoBias evaluations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


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
DEFAULT_MODEL_GROUPS: dict[str, ModelGroup] = {
    "3b": ModelGroup(
        name="3b",
        description="~3B parameter class models",
        models=[
            ModelSpec("qwen/qwen3-1.7b", "Qwen3 1.7B", "CN", 1.7, "Qwen"),
            ModelSpec("meta-llama/llama-3.2-3b-instruct", "Llama 3.2 3B", "US", 3.0, "Llama"),
            ModelSpec("microsoft/phi-3.5-mini-128k-instruct", "Phi-3.5 Mini", "US", 3.8, "Phi"),
            ModelSpec("mistralai/ministral-3b", "Ministral 3B", "EU", 3.0, "Mistral"),
        ],
    ),
    "7b": ModelGroup(
        name="7b",
        description="~7-8B parameter class models",
        models=[
            ModelSpec("qwen/qwen3-8b", "Qwen3 8B", "CN", 8.0, "Qwen"),
            ModelSpec("deepseek/deepseek-r1-distill-qwen-7b", "DeepSeek R1 Distill 7B", "CN", 7.0, "DeepSeek"),
            ModelSpec("meta-llama/llama-3.1-8b-instruct", "Llama 3.1 8B", "US", 8.0, "Llama"),
            ModelSpec("google/gemma-3-4b-it", "Gemma 3 4B", "US", 4.0, "Gemma"),
            ModelSpec("microsoft/phi-4-mini-instruct", "Phi-4 Mini", "US", 3.8, "Phi"),
            ModelSpec("mistralai/ministral-8b", "Ministral 8B", "EU", 8.0, "Mistral"),
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


@dataclass
class EvalConfig:
    """Full configuration for an evaluation run."""

    models: list[ModelSpec]
    categories: list[str]
    cases_per_category: int = 8
    orchestrator_model: str = "anthropic/claude-sonnet-4"
    output_dir: Path = Path("reports")
    group_name: Optional[str] = None
    dry_run: bool = False

    def __post_init__(self):
        for cat in self.categories:
            if cat not in ALL_CATEGORIES:
                raise ValueError(f"Unknown category: {cat}. Valid: {ALL_CATEGORIES}")

    def run_dir(self) -> Path:
        """Create and return the directory for this run."""
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
            "orchestrator_model": self.orchestrator_model,
            "group_name": self.group_name,
        }


def get_orchestrator_model() -> str:
    return os.environ.get("GEOBIAS_ORCHESTRATOR", "anthropic/claude-sonnet-4")


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
