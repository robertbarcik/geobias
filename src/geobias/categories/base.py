"""Base class for test categories."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class TestCase:
    """A single test prompt with metadata."""

    id: str
    category: str
    prompt: str
    system_prompt: str = "You are a helpful assistant."
    target_region: Optional[str] = None  # Which region this tests sensitivity to
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelResponse:
    """A model's response to a test case."""

    test_case_id: str
    model_id: str
    response: str
    usage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Evaluation:
    """An evaluator's assessment of a model response."""

    test_case_id: str
    model_id: str
    scores: dict  # category-specific scores
    classification: Optional[str] = None  # e.g. "answered", "hedged", "deflected", "refused"
    justification: str = ""
    highlights: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)  # evaluator info, agreement stats

    def to_dict(self) -> dict:
        return asdict(self)


class TestCategory(abc.ABC):
    """Base class for test categories. Each category generates prompts and evaluates responses."""

    name: str
    description: str

    @abc.abstractmethod
    def generation_prompt(self, cases_per_category: int) -> list[dict]:
        """Return the messages to send to the orchestrator to generate test cases."""
        ...

    @abc.abstractmethod
    def parse_test_cases(self, orchestrator_response: str) -> list[TestCase]:
        """Parse the orchestrator's response into structured test cases."""
        ...

    @abc.abstractmethod
    def evaluation_prompt(self, test_case: TestCase, response: str) -> list[dict]:
        """Return messages to send to the orchestrator to evaluate a model's response."""
        ...

    @abc.abstractmethod
    def parse_evaluation(self, test_case_id: str, model_id: str, orchestrator_response: str) -> Evaluation:
        """Parse the orchestrator's evaluation response."""
        ...
