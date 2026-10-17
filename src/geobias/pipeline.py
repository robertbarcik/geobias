"""Core evaluation pipeline — orchestrates test generation, model querying, and evaluation."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from .categories.base import (
    EXCLUDED_SCORE_KEYS,
    Evaluation,
    ModelResponse,
    TestCase,
    numeric_scores,
)
from .categories.registry import get_categories
from .config import EvalConfig, ModelSpec
from .llm import LLMClient

logger = logging.getLogger(__name__)

# Token budget for subject-model responses. Reasoning models spend hidden thinking
# tokens inside this budget; at 2048 (the March 2026 value) 69 of 704 October responses
# came back empty and 204 more were cut off mid-answer, which evaluators would read as
# refusals. 8192 still left 26 capped (Qwen3.8 27B: 14); 16384 leaves room for both.
RESPONSE_MAX_TOKENS = 16384

# Regions used for the own-region vs other-region asymmetry split. Cases whose
# target_region is neutral/other/None are excluded from both sides (matching
# the report's origin x topic matrix logic).
CORE_REGIONS = ("CN", "US", "EU")


class EvaluationPipeline:
    """Runs the full GeoBias evaluation pipeline.

    Uses a panel of evaluator models from different origins (CN, US, EU) to
    cross-validate assessments. Each evaluator judges every response independently.
    The final scores are consensus-based: where evaluators agree, confidence is high;
    where they disagree, the disagreement itself is informative data about evaluator bias.
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self.run_dir = config.run_dir()
        self.cache_dir = self.run_dir / ".cache"
        self.client = LLMClient(cache_dir=self.cache_dir)

        # Storage
        self.test_cases: dict[str, list[TestCase]] = {}  # category -> cases
        self.responses: dict[str, list[ModelResponse]] = {}  # model_id -> responses
        # evaluations keyed by evaluator_id -> list of Evaluation
        self.evaluations_by_evaluator: dict[str, list[Evaluation]] = {}
        # consensus evaluations (merged across evaluators)
        self.evaluations: list[Evaluation] = []

    async def run(self) -> Path:
        """Execute the full pipeline. Returns path to the run directory."""
        self._save_config()

        categories = get_categories(self.config.categories)
        evaluators = self.config.evaluators

        # Stage 1: Generate test cases (use first evaluator for generation)
        logger.info("Stage 1: Generating test cases...")
        generator_model = evaluators[0].id
        for cat in categories:
            cases = await self._generate_test_cases(cat, generator_model)
            self.test_cases[cat.name] = cases
            logger.info(f"  {cat.name}: {len(cases)} test cases generated")
        self._save_test_cases()

        # Stage 2: Collect model responses
        logger.info("Stage 2: Collecting model responses...")
        all_cases = [c for cases in self.test_cases.values() for c in cases]
        # All subject models are collected concurrently; the client semaphore and
        # rate limiter bound the number of requests in flight.
        async def collect_one(model: ModelSpec) -> None:
            model_responses = await self._collect_responses(model, all_cases)
            self.responses[model.id] = model_responses
            logger.info(f"  {model.name}: {len(model_responses)} responses collected")
            self._save_responses(model)

        await asyncio.gather(*[collect_one(m) for m in self.config.models])

        # Stage 3: Evaluate responses with each evaluator
        logger.info(f"Stage 3: Evaluating responses with {len(evaluators)} evaluator(s)...")

        async def evaluate_with(evaluator: ModelSpec) -> None:
            logger.info(f"  Evaluator: {evaluator.name} [{evaluator.origin}]")
            evaluator_evals = []
            for cat in categories:
                cases = self.test_cases[cat.name]
                for model in self.config.models:
                    model_responses = {r.test_case_id: r for r in self.responses[model.id]}
                    pending = [(case, model_responses.get(case.id)) for case in cases]
                    pending = [(case, resp) for case, resp in pending if resp]
                    # Evaluate a block concurrently (bounded by the client semaphore);
                    # _evaluate_response never raises, so plain gather keeps the order.
                    batch_size = 16
                    for i in range(0, len(pending), batch_size):
                        batch = pending[i : i + batch_size]
                        results = await asyncio.gather(*[
                            self._evaluate_response(cat, case, model, resp.response, evaluator)
                            for case, resp in batch
                        ])
                        evaluator_evals.extend(results)
                    logger.info(f"    [{evaluator.name}] {cat.name} × {model.name}: done")
            self.evaluations_by_evaluator[evaluator.id] = evaluator_evals

        # The three evaluators run side by side; each keeps its own ordered list.
        await asyncio.gather(*[evaluate_with(e) for e in evaluators])

        # Stage 3b: Compute consensus evaluations
        logger.info("Stage 3b: Computing evaluator consensus...")
        self.evaluations = self._merge_evaluations()
        self._save_evaluations()

        # Stage 4: Compute metrics
        logger.info("Stage 4: Computing metrics...")
        metrics = self._compute_metrics()
        self._save_metrics(metrics)

        logger.info(f"Pipeline complete. Results in: {self.run_dir}")
        return self.run_dir

    async def _generate_test_cases(self, category, generator_model: str) -> list[TestCase]:
        """Use the first evaluator to generate test cases for a category."""
        cache_file = self.run_dir / "test_cases.json"
        if cache_file.exists():
            existing = json.loads(cache_file.read_text())
            if category.name in existing:
                return [TestCase(**tc) for tc in existing[category.name]]

        messages = category.generation_prompt(self.config.cases_per_category)
        response = await self.client.complete(
            model=generator_model,
            messages=messages,
            temperature=0.7,
            max_tokens=8192,
        )
        return category.parse_test_cases(response.content)

    async def _collect_responses(self, model: ModelSpec, cases: list[TestCase]) -> list[ModelResponse]:
        """Collect responses from a subject model for all test cases."""
        resp_file = self.run_dir / "responses" / f"{model.safe_filename()}.json"
        if resp_file.exists():
            existing = json.loads(resp_file.read_text())
            # On resume, rows saved as transport errors (e.g. upstream 429) are retried.
            existing = [r for r in existing if not str(r.get("response", "")).startswith("[ERROR")]
            cached_ids = {r["test_case_id"] for r in existing}
            cached = [ModelResponse(**r) for r in existing]
        else:
            cached_ids = set()
            cached = []

        remaining = [c for c in cases if c.id not in cached_ids]
        if not remaining:
            return cached

        batch_size = 10
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

        return cached + new_responses

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
            max_tokens=RESPONSE_MAX_TOKENS,
        )
        return ModelResponse(
            test_case_id=case.id,
            model_id=model.id,
            response=response.content,
            usage=response.usage,
        )

    async def _evaluate_response(
        self, category, case: TestCase, model: ModelSpec, response_text: str, evaluator: ModelSpec
    ) -> Evaluation:
        """Use a specific evaluator to evaluate a model's response."""
        # If the response is empty, tell the evaluator explicitly so it doesn't hallucinate
        if not response_text.strip():
            response_text = "[EMPTY RESPONSE — the model returned no text. Score all criteria as 0.]"
        messages = category.evaluation_prompt(case, response_text)
        try:
            resp = await self.client.complete(
                model=evaluator.id,
                messages=messages,
                temperature=0.0,
                max_tokens=4096,
            )
            ev = category.parse_evaluation(case.id, model.id, resp.content)
            ev.metadata = {"evaluator_id": evaluator.id, "evaluator_origin": evaluator.origin}
            return ev
        except Exception as e:
            logger.warning(f"Evaluation parse error ({evaluator.name}) for {model.id}/{case.id}: {e}")
            return Evaluation(
                test_case_id=case.id,
                model_id=model.id,
                scores={},
                justification=f"[Evaluation error from {evaluator.name}: {e}]",
                metadata={"evaluator_id": evaluator.id, "evaluator_origin": evaluator.origin},
            )

    def _merge_evaluations(self) -> list[Evaluation]:
        """Merge evaluations from multiple evaluators into consensus evaluations.

        For each (test_case_id, model_id) pair, average numeric scores across evaluators
        and track per-evaluator details for disagreement analysis.
        """
        # Group by (test_case_id, model_id)
        grouped: dict[tuple[str, str], list[Evaluation]] = {}
        for evaluator_id, evals in self.evaluations_by_evaluator.items():
            for ev in evals:
                key = (ev.test_case_id, ev.model_id)
                grouped.setdefault(key, []).append(ev)

        consensus = []
        for (tc_id, model_id), evals in grouped.items():
            # Average numeric scores
            all_score_keys = set()
            for ev in evals:
                all_score_keys.update(ev.scores.keys())

            merged_scores = {}
            for key in all_score_keys:
                if key in EXCLUDED_SCORE_KEYS:
                    continue  # off-scale signals are merged into scores_extra below
                values = [ev.scores[key] for ev in evals if key in ev.scores and isinstance(ev.scores[key], (int, float))]
                if values:
                    merged_scores[key] = round(sum(values) / len(values), 2)

            # Merge off-scale signals (e.g. sentiment_score) separately
            extra_values: dict[str, list] = {}
            for ev in evals:
                for k, v in (getattr(ev, "scores_extra", {}) or {}).items():
                    if isinstance(v, (int, float)):
                        extra_values.setdefault(k, []).append(v)
                for k in EXCLUDED_SCORE_KEYS:
                    v = ev.scores.get(k)
                    if isinstance(v, (int, float)):
                        extra_values.setdefault(k, []).append(v)
            merged_extra = {k: round(sum(vals) / len(vals), 2) for k, vals in extra_values.items()}

            # Majority vote for classification
            classifications = [ev.classification for ev in evals if ev.classification]
            if classifications:
                majority_cls = max(set(classifications), key=classifications.count)
            else:
                majority_cls = None

            # Collect all justifications
            justifications = [ev.justification for ev in evals if ev.justification]
            merged_justification = " | ".join(justifications) if justifications else ""

            # All highlights
            all_highlights = []
            for ev in evals:
                all_highlights.extend(ev.highlights)

            # Per-evaluator breakdown for disagreement analysis
            per_evaluator = {}
            for ev in evals:
                meta = getattr(ev, "metadata", {}) or {}
                eid = meta.get("evaluator_id", "unknown")
                per_evaluator[eid] = {
                    "scores": ev.scores,
                    "classification": ev.classification,
                    "origin": meta.get("evaluator_origin", "unknown"),
                }

            # Compute evaluator agreement (std dev of overall scores)
            score_lists = {}
            for ev in evals:
                for k, v in ev.scores.items():
                    if k not in EXCLUDED_SCORE_KEYS and isinstance(v, (int, float)):
                        score_lists.setdefault(k, []).append(v)
            agreement = {}
            for k, vals in score_lists.items():
                if len(vals) > 1:
                    mean = sum(vals) / len(vals)
                    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
                    agreement[k] = round(variance ** 0.5, 2)  # std dev

            consensus.append(Evaluation(
                test_case_id=tc_id,
                model_id=model_id,
                scores=merged_scores,
                classification=majority_cls,
                justification=merged_justification,
                highlights=all_highlights,
                scores_extra=merged_extra,
                metadata={
                    "per_evaluator": per_evaluator,
                    "evaluator_agreement_stddev": agreement,
                    "num_evaluators": len(evals),
                },
            ))

        return consensus

    def _compute_metrics(self) -> dict:
        """Compute comparative metrics across models."""
        metrics = {
            "per_model": {},
            "per_category": {},
            "refusal_matrix": {},
            "bias_asymmetry": {},
            "evaluator_disagreement": {},
        }

        # Build test case lookup for efficient region mapping
        tc_lookup: dict[str, TestCase] = {}
        for cases in self.test_cases.values():
            for tc in cases:
                tc_lookup[tc.id] = tc

        # Per-model aggregates
        for model in self.config.models:
            model_evals = [e for e in self.evaluations if e.model_id == model.id]
            if not model_evals:
                continue

            all_scores = []
            for ev in model_evals:
                all_scores.extend(numeric_scores(ev.scores))

            metrics["per_model"][model.id] = {
                "name": model.name,
                "origin": model.origin,
                "total_evaluations": len(model_evals),
                "avg_score": round(sum(all_scores) / len(all_scores), 2) if all_scores else 0,
                "classifications": {},
            }

            for ev in model_evals:
                cls = ev.classification or "unknown"
                metrics["per_model"][model.id]["classifications"][cls] = (
                    metrics["per_model"][model.id]["classifications"].get(cls, 0) + 1
                )

        # Per-category aggregates
        for cat_name in self.config.categories:
            cat_case_ids = {tc.id for tc in self.test_cases.get(cat_name, [])}
            cat_evals = [e for e in self.evaluations if e.test_case_id in cat_case_ids]
            per_model = {}
            for model in self.config.models:
                model_cat_evals = [e for e in cat_evals if e.model_id == model.id]
                if not model_cat_evals:
                    continue
                scores = []
                for ev in model_cat_evals:
                    scores.extend(numeric_scores(ev.scores))
                per_model[model.id] = {
                    "avg_score": round(sum(scores) / len(scores), 2) if scores else 0,
                    "num_evaluations": len(model_cat_evals),
                }
            metrics["per_category"][cat_name] = per_model

        # Refusal matrix
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

        # Bias asymmetry (own-region vs other-region topics). Only cases whose
        # target_region is one of CORE_REGIONS enter the split; neutral/other
        # region cases are excluded from both sides (matching the report matrix).
        for model in self.config.models:
            if model.origin not in CORE_REGIONS:
                continue
            own_region_evals = []
            other_region_evals = []
            for ev in self.evaluations:
                if ev.model_id != model.id:
                    continue
                tc = tc_lookup.get(ev.test_case_id)
                if not tc or tc.target_region not in CORE_REGIONS:
                    continue
                if tc.target_region == model.origin:
                    own_region_evals.append(ev)
                else:
                    other_region_evals.append(ev)

            def avg_scores(evals):
                all_s = []
                for ev in evals:
                    all_s.extend(numeric_scores(ev.scores))
                return round(sum(all_s) / len(all_s), 2) if all_s else 0

            metrics["bias_asymmetry"][model.id] = {
                "own_region_avg": avg_scores(own_region_evals),
                "other_region_avg": avg_scores(other_region_evals),
                "own_region_count": len(own_region_evals),
                "other_region_count": len(other_region_evals),
                "asymmetry": round(avg_scores(other_region_evals) - avg_scores(own_region_evals), 2),
            }

        # Evaluator disagreement analysis
        for evaluator_id, evals in self.evaluations_by_evaluator.items():
            evaluator_origin = "unknown"
            for e in self.config.evaluators:
                if e.id == evaluator_id:
                    evaluator_origin = e.origin
                    break

            if evaluator_origin == "unknown":
                logger.warning(
                    f"Evaluator {evaluator_id} has unknown origin; skipping the "
                    f"own-origin leniency metric for it (cannot define own vs other origin)."
                )
                metrics["evaluator_disagreement"][evaluator_id] = {
                    "origin": "unknown",
                    "skipped": True,
                    "note": "Own-origin leniency not computable for an evaluator with unknown origin.",
                }
                continue

            # Does this evaluator score models from its own origin higher?
            own_origin_scores = []
            other_origin_scores = []
            for ev in evals:
                subject_model = next((m for m in self.config.models if m.id == ev.model_id), None)
                if not subject_model:
                    continue
                numeric = numeric_scores(ev.scores)
                if subject_model.origin == evaluator_origin:
                    own_origin_scores.extend(numeric)
                else:
                    other_origin_scores.extend(numeric)

            own_avg = round(sum(own_origin_scores) / len(own_origin_scores), 2) if own_origin_scores else 0
            other_avg = round(sum(other_origin_scores) / len(other_origin_scores), 2) if other_origin_scores else 0

            metrics["evaluator_disagreement"][evaluator_id] = {
                "origin": evaluator_origin,
                "avg_score_own_origin_models": own_avg,
                "avg_score_other_origin_models": other_avg,
                "own_origin_leniency": round(own_avg - other_avg, 2),
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
        # Save consensus evaluations
        path = self.run_dir / "evaluations.json"
        data = [e.to_dict() for e in self.evaluations]
        path.write_text(json.dumps(data, indent=2))

        # Save per-evaluator raw evaluations
        eval_dir = self.run_dir / "evaluations_per_evaluator"
        eval_dir.mkdir(exist_ok=True)
        for evaluator_id, evals in self.evaluations_by_evaluator.items():
            safe_name = evaluator_id.replace("/", "--")
            path = eval_dir / f"{safe_name}.json"
            path.write_text(json.dumps([e.to_dict() for e in evals], indent=2))

    def _save_metrics(self, metrics: dict):
        path = self.run_dir / "metrics.json"
        path.write_text(json.dumps(metrics, indent=2))
