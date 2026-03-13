"""Core evaluation pipeline — orchestrates test generation, model querying, and evaluation."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from .categories.base import TestCase, ModelResponse, Evaluation
from .categories.registry import get_categories
from .config import EvalConfig, ModelSpec
from .llm import LLMClient

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Runs the full GeoBias evaluation pipeline."""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.run_dir = config.run_dir()
        self.cache_dir = self.run_dir / ".cache"
        self.client = LLMClient(cache_dir=self.cache_dir)

        # Storage
        self.test_cases: dict[str, list[TestCase]] = {}  # category -> cases
        self.responses: dict[str, list[ModelResponse]] = {}  # model_id -> responses
        self.evaluations: list[Evaluation] = []

    async def run(self) -> Path:
        """Execute the full pipeline. Returns path to the run directory."""
        self._save_config()

        categories = get_categories(self.config.categories)

        # Stage 1: Generate test cases
        logger.info("Stage 1: Generating test cases...")
        for cat in categories:
            cases = await self._generate_test_cases(cat)
            self.test_cases[cat.name] = cases
            logger.info(f"  {cat.name}: {len(cases)} test cases generated")
        self._save_test_cases()

        # Stage 2: Collect model responses
        logger.info("Stage 2: Collecting model responses...")
        all_cases = [c for cases in self.test_cases.values() for c in cases]
        for model in self.config.models:
            model_responses = await self._collect_responses(model, all_cases)
            self.responses[model.id] = model_responses
            logger.info(f"  {model.name}: {len(model_responses)} responses collected")
            self._save_responses(model)

        # Stage 3: Evaluate responses
        logger.info("Stage 3: Evaluating responses...")
        for cat in categories:
            cases = self.test_cases[cat.name]
            for model in self.config.models:
                model_responses = {r.test_case_id: r for r in self.responses[model.id]}
                for case in cases:
                    resp = model_responses.get(case.id)
                    if not resp:
                        continue
                    evaluation = await self._evaluate_response(cat, case, model, resp.response)
                    self.evaluations.append(evaluation)
                logger.info(f"  {cat.name} × {model.name}: evaluated")
        self._save_evaluations()

        # Stage 4: Compute metrics
        logger.info("Stage 4: Computing metrics...")
        metrics = self._compute_metrics()
        self._save_metrics(metrics)

        logger.info(f"Pipeline complete. Results in: {self.run_dir}")
        return self.run_dir

    async def _generate_test_cases(self, category) -> list[TestCase]:
        """Use the orchestrator to generate test cases for a category."""
        # Check for cached test cases
        cache_file = self.run_dir / "test_cases.json"
        if cache_file.exists():
            existing = json.loads(cache_file.read_text())
            if category.name in existing:
                return [TestCase(**tc) for tc in existing[category.name]]

        messages = category.generation_prompt(self.config.cases_per_category)
        response = await self.client.complete(
            model=self.config.orchestrator_model,
            messages=messages,
            temperature=0.7,
            max_tokens=8192,
        )
        return category.parse_test_cases(response.content)

    async def _collect_responses(self, model: ModelSpec, cases: list[TestCase]) -> list[ModelResponse]:
        """Collect responses from a subject model for all test cases."""
        # Check for cached responses
        resp_file = self.run_dir / "responses" / f"{model.safe_filename()}.json"
        if resp_file.exists():
            existing = json.loads(resp_file.read_text())
            cached_ids = {r["test_case_id"] for r in existing}
            cached = [ModelResponse(**r) for r in existing]
        else:
            cached_ids = set()
            cached = []

        remaining = [c for c in cases if c.id not in cached_ids]
        if not remaining:
            return cached

        # Process in batches
        batch_size = 5
        new_responses = []
        for i in range(0, len(remaining), batch_size):
            batch = remaining[i : i + batch_size]
            tasks = [self._get_model_response(model, case) for case in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for case, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.warning(f"Error getting response from {model.id} for {case.id}: {result}")
                    new_responses.append(ModelResponse(
                        test_case_id=case.id,
                        model_id=model.id,
                        response=f"[ERROR: {result}]",
                    ))
                else:
                    new_responses.append(result)

        all_responses = cached + new_responses
        return all_responses

    async def _get_model_response(self, model: ModelSpec, case: TestCase) -> ModelResponse:
        """Get a single model's response to a test case."""
        messages = [
            {"role": "system", "content": case.system_prompt},
            {"role": "user", "content": case.prompt},
        ]
        response = await self.client.complete(
            model=model.id,
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
        )
        return ModelResponse(
            test_case_id=case.id,
            model_id=model.id,
            response=response.content,
            usage=response.usage,
        )

    async def _evaluate_response(self, category, case: TestCase, model: ModelSpec, response_text: str) -> Evaluation:
        """Use the orchestrator to evaluate a model's response."""
        messages = category.evaluation_prompt(case, response_text)
        try:
            resp = await self.client.complete(
                model=self.config.orchestrator_model,
                messages=messages,
                temperature=0.0,
                max_tokens=4096,
            )
            return category.parse_evaluation(case.id, model.id, resp.content)
        except Exception as e:
            logger.warning(f"Evaluation parse error for {model.id}/{case.id}: {e}")
            return Evaluation(
                test_case_id=case.id,
                model_id=model.id,
                scores={},
                justification=f"[Evaluation error: {e}]",
            )

    def _compute_metrics(self) -> dict:
        """Compute comparative metrics across models."""
        metrics = {
            "per_model": {},
            "per_category": {},
            "refusal_matrix": {},
            "bias_asymmetry": {},
        }

        # Per-model aggregates
        for model in self.config.models:
            model_evals = [e for e in self.evaluations if e.model_id == model.id]
            if not model_evals:
                continue

            all_scores = []
            for ev in model_evals:
                all_scores.extend(ev.scores.values())
            numeric_scores = [s for s in all_scores if isinstance(s, (int, float))]

            metrics["per_model"][model.id] = {
                "name": model.name,
                "origin": model.origin,
                "total_evaluations": len(model_evals),
                "avg_score": round(sum(numeric_scores) / len(numeric_scores), 2) if numeric_scores else 0,
                "classifications": {},
            }

            # Classification distribution
            for ev in model_evals:
                cls = ev.classification or "unknown"
                metrics["per_model"][model.id]["classifications"][cls] = (
                    metrics["per_model"][model.id]["classifications"].get(cls, 0) + 1
                )

        # Per-category aggregates
        for cat_name in self.config.categories:
            cat_evals = [e for e in self.evaluations if any(
                tc.id == e.test_case_id
                for cases in self.test_cases.values()
                for tc in cases
                if tc.category == cat_name
            )]
            per_model = {}
            for model in self.config.models:
                model_cat_evals = [e for e in cat_evals if e.model_id == model.id]
                if not model_cat_evals:
                    continue
                scores = []
                for ev in model_cat_evals:
                    scores.extend(v for v in ev.scores.values() if isinstance(v, (int, float)))
                per_model[model.id] = {
                    "avg_score": round(sum(scores) / len(scores), 2) if scores else 0,
                    "num_evaluations": len(model_cat_evals),
                }
            metrics["per_category"][cat_name] = per_model

        # Refusal matrix (for refusal_asymmetry category)
        if "refusal_asymmetry" in self.config.categories:
            cases_by_region = {}
            for tc in self.test_cases.get("refusal_asymmetry", []):
                region = tc.target_region or "neutral"
                cases_by_region.setdefault(region, []).append(tc.id)

            for model in self.config.models:
                model_matrix = {}
                for region, case_ids in cases_by_region.items():
                    region_evals = [
                        e for e in self.evaluations
                        if e.model_id == model.id and e.test_case_id in case_ids
                    ]
                    classification_counts = {}
                    for ev in region_evals:
                        cls = ev.classification or "unknown"
                        classification_counts[cls] = classification_counts.get(cls, 0) + 1
                    model_matrix[region] = classification_counts
                metrics["refusal_matrix"][model.id] = model_matrix

        # Bias asymmetry — compare how each model treats its "own" region vs others
        for model in self.config.models:
            if model.origin == "unknown":
                continue
            own_region_evals = []
            other_region_evals = []
            for ev in self.evaluations:
                if ev.model_id != model.id:
                    continue
                # Find the test case to get its target_region
                tc = None
                for cases in self.test_cases.values():
                    for c in cases:
                        if c.id == ev.test_case_id:
                            tc = c
                            break
                if not tc or not tc.target_region:
                    continue
                if tc.target_region == model.origin:
                    own_region_evals.append(ev)
                else:
                    other_region_evals.append(ev)

            def avg_scores(evals):
                all_s = []
                for ev in evals:
                    all_s.extend(v for v in ev.scores.values() if isinstance(v, (int, float)))
                return round(sum(all_s) / len(all_s), 2) if all_s else 0

            metrics["bias_asymmetry"][model.id] = {
                "own_region_avg": avg_scores(own_region_evals),
                "other_region_avg": avg_scores(other_region_evals),
                "own_region_count": len(own_region_evals),
                "other_region_count": len(other_region_evals),
                "asymmetry": round(avg_scores(other_region_evals) - avg_scores(own_region_evals), 2),
            }

        return metrics

    # --- Persistence ---

    def _save_config(self):
        path = self.run_dir / "config.json"
        path.write_text(json.dumps(self.config.to_dict(), indent=2))

    def _save_test_cases(self):
        path = self.run_dir / "test_cases.json"
        data = {cat: [tc.to_dict() for tc in cases] for cat, cases in self.test_cases.items()}
        path.write_text(json.dumps(data, indent=2))

    def _save_responses(self, model: ModelSpec):
        resp_dir = self.run_dir / "responses"
        resp_dir.mkdir(exist_ok=True)
        path = resp_dir / f"{model.safe_filename()}.json"
        data = [r.to_dict() for r in self.responses[model.id]]
        path.write_text(json.dumps(data, indent=2))

    def _save_evaluations(self):
        path = self.run_dir / "evaluations.json"
        data = [e.to_dict() for e in self.evaluations]
        path.write_text(json.dumps(data, indent=2))

    def _save_metrics(self, metrics: dict):
        path = self.run_dir / "metrics.json"
        path.write_text(json.dumps(metrics, indent=2))
