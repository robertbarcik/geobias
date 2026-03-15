"""HTML report generation from evaluation results."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .categories.base import TestCase, Evaluation
from .categories.prompts import ORCHESTRATOR_SYSTEM_PROMPT, TOPIC_REGIONS
from .config import ModelSpec

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"

CATEGORY_DESCRIPTIONS = {
    "narrative_framing": "How models frame geopolitical events — detecting one-sided narratives and loaded language.",
    "refusal_asymmetry": "Patterns of refusals and deflections across topics sensitive to different regions.",
    "entity_sentiment": "Sentiment bias in descriptions of political leaders, institutions, and countries.",
    "language_quality": "Language quality across multiple languages, detecting training data bias.",
    "regulatory_values": "Whether models favor their origin region's regulatory stance.",
    "self_awareness": "Model self-awareness about biases, limitations, and training.",
    "historical_narrative": "Control category — historically settled topics that should be handled similarly by all models.",
}


def load_run_data(run_dir: Path) -> dict:
    """Load all data from a completed run directory."""
    data = {}

    config_path = run_dir / "config.json"
    if config_path.exists():
        data["config"] = json.loads(config_path.read_text())

    tc_path = run_dir / "test_cases.json"
    if tc_path.exists():
        data["test_cases"] = json.loads(tc_path.read_text())

    eval_path = run_dir / "evaluations.json"
    if eval_path.exists():
        data["evaluations"] = json.loads(eval_path.read_text())

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        data["metrics"] = json.loads(metrics_path.read_text())

    # Load responses
    data["responses"] = {}
    resp_dir = run_dir / "responses"
    if resp_dir.exists():
        for f in resp_dir.glob("*.json"):
            model_key = f.stem
            data["responses"][model_key] = json.loads(f.read_text())

    # Load per-evaluator evaluations
    data["per_evaluator"] = {}
    eval_dir = run_dir / "evaluations_per_evaluator"
    if eval_dir.exists():
        for f in eval_dir.glob("*.json"):
            evaluator_key = f.stem
            data["per_evaluator"][evaluator_key] = json.loads(f.read_text())

    return data


def generate_report(run_dir: Path) -> Path:
    """Generate an HTML report from run data. Returns path to the report file."""
    data = load_run_data(run_dir)

    if not data.get("config"):
        raise FileNotFoundError(f"No config.json found in {run_dir}")

    models = [ModelSpec(**m) for m in data["config"]["models"]]
    categories = data["config"]["categories"]
    metrics = data.get("metrics", {})
    evaluations = data.get("evaluations", [])
    test_cases = data.get("test_cases", {})
    responses = data.get("responses", {})
    per_evaluator = data.get("per_evaluator", {})
    evaluators = [ModelSpec(**e) for e in data["config"].get("evaluators", [])]

    # Build lookups
    response_lookup = {}
    for model_key, resp_list in responses.items():
        for r in resp_list:
            response_lookup[(r["test_case_id"], r["model_id"])] = r["response"]

    eval_lookup = {}
    for ev in evaluations:
        eval_lookup[(ev["test_case_id"], ev["model_id"])] = ev

    # Build per-evaluator justification lookup: (test_case_id, model_id, evaluator_key) -> justification
    per_eval_justifications = {}
    for eval_key, evals in per_evaluator.items():
        for ev in evals:
            per_eval_justifications[(ev["test_case_id"], ev["model_id"], eval_key)] = {
                "justification": ev.get("justification", ""),
                "scores": ev.get("scores", {}),
                "classification": ev.get("classification"),
            }

    tc_lookup = {}
    for cat_name, cases in test_cases.items():
        for c in cases:
            tc_lookup[c["id"]] = {**c, "category": cat_name}

    total_responses = sum(len(resp_list) for resp_list in responses.values())
    is_partial = len(evaluations) == 0 and total_responses > 0

    # --- Compute all report sections ---
    summary = _compute_executive_summary(models, metrics, evaluations, test_cases, total_responses)
    origin_topic_matrix = _build_origin_topic_matrix(models, evaluations, tc_lookup)
    refusal_heatmap = _build_refusal_heatmap(models, metrics)
    evaluator_analysis = _build_evaluator_analysis(evaluators, per_evaluator, models, metrics)
    category_details = []
    for cat_name in categories:
        cat_cases = test_cases.get(cat_name, [])
        cat_data = _build_category_detail(cat_name, cat_cases, models, evaluations, response_lookup, evaluators, per_eval_justifications)
        category_details.append(cat_data)
    model_profiles = _build_model_profiles(models, metrics, evaluations, per_evaluator, tc_lookup)
    extreme_cases = _build_extreme_cases(models, evaluations, tc_lookup, response_lookup, evaluators, per_eval_justifications)
    commentary = _generate_commentary(models, metrics, evaluations, tc_lookup, response_lookup, origin_topic_matrix)
    empty_response_analysis = _build_empty_response_analysis(models, responses, tc_lookup)
    deflection_analysis = _build_deflection_analysis(models, evaluations, tc_lookup, response_lookup)

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html")

    html = template.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        run_dir=str(run_dir),
        models=models,
        categories=categories,
        evaluators=evaluators,
        summary=summary,
        origin_topic_matrix=origin_topic_matrix,
        refusal_heatmap=refusal_heatmap,
        evaluator_analysis=evaluator_analysis,
        category_details=category_details,
        model_profiles=model_profiles,
        extreme_cases=extreme_cases,
        commentary=commentary,
        empty_response_analysis=empty_response_analysis,
        deflection_analysis=deflection_analysis,
        metrics=metrics,
        orchestrator_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        topic_regions=TOPIC_REGIONS,
        raw_data=json.dumps(data, indent=2, default=str),
        is_partial=is_partial,
        total_responses=total_responses,
    )

    report_path = run_dir / "report.html"
    report_path.write_text(html)
    logger.info(f"Report generated: {report_path}")
    return report_path


def _compute_executive_summary(models, metrics, evaluations, test_cases, total_responses) -> dict:
    per_model = metrics.get("per_model", {})
    asymmetry = metrics.get("bias_asymmetry", {})

    model_summaries = []
    for model in models:
        pm = per_model.get(model.id, {})
        asym = asymmetry.get(model.id, {})
        model_summaries.append({
            "name": model.name,
            "id": model.id,
            "origin": model.origin,
            "params_b": model.params_b,
            "avg_score": pm.get("avg_score", 0),
            "asymmetry_score": asym.get("asymmetry", 0),
            "own_region_avg": asym.get("own_region_avg", 0),
            "other_region_avg": asym.get("other_region_avg", 0),
            "total_evaluations": pm.get("total_evaluations", 0),
        })

    model_summaries.sort(key=lambda x: abs(x["asymmetry_score"]), reverse=True)

    total_test_cases = sum(len(cases) for cases in test_cases.values())
    total_evaluations = len(evaluations)

    return {
        "total_models": len(models),
        "total_test_cases": total_test_cases,
        "total_evaluations": total_evaluations,
        "total_responses": total_responses,
        "model_summaries": model_summaries,
        "is_partial": total_evaluations == 0 and total_test_cases > 0,
    }


def _build_origin_topic_matrix(models, evaluations, tc_lookup) -> dict:
    """Build the origin × topic region score matrix — the centerpiece visualization."""
    if not evaluations:
        return {}

    # Use same order for both axes so the diagonal is visually correct
    regions = ["CN", "US", "EU"]
    origins = regions  # rows must match column order for diagonal

    # Collect scores per (subject_origin, target_region)
    cells = {}
    for ev in evaluations:
        model = next((m for m in models if m.id == ev["model_id"]), None)
        tc = tc_lookup.get(ev["test_case_id"])
        if not model or not tc:
            continue
        target = tc.get("target_region")
        if not target or target not in regions:
            continue

        key = (model.origin, target)
        scores = [v for v in ev.get("scores", {}).values() if isinstance(v, (int, float))]
        cells.setdefault(key, []).extend(scores)

    # Also collect per-model scores for the detailed view
    per_model = {}
    for ev in evaluations:
        tc = tc_lookup.get(ev["test_case_id"])
        if not tc:
            continue
        target = tc.get("target_region")
        if not target or target not in regions:
            continue
        model_id = ev["model_id"]
        key = (model_id, target)
        scores = [v for v in ev.get("scores", {}).values() if isinstance(v, (int, float))]
        per_model.setdefault(key, []).extend(scores)

    matrix = {
        "origins": origins,
        "regions": regions,
        "cells": {},
        "per_model": {},
    }

    for (origin, region), scores in cells.items():
        matrix["cells"][f"{origin}_{region}"] = {
            "avg": round(mean(scores), 2) if scores else 0,
            "count": len(scores),
            "n_evals": len(scores),
        }

    for (model_id, region), scores in per_model.items():
        matrix["per_model"][f"{model_id}_{region}"] = {
            "avg": round(mean(scores), 2) if scores else 0,
            "count": len(scores),
        }

    return matrix


def _build_refusal_heatmap(models, metrics) -> list[dict]:
    refusal_matrix = metrics.get("refusal_matrix", {})
    rows = []
    for model in models:
        matrix = refusal_matrix.get(model.id, {})
        row = {
            "model_name": model.name,
            "model_id": model.id,
            "model_origin": model.origin,
            "regions": {},
        }
        for region in ["CN", "US", "EU", "neutral"]:
            counts = matrix.get(region, {})
            total = sum(counts.values()) if counts else 0
            row["regions"][region] = {
                "counts": counts,
                "total": total,
                "answered_pct": round(counts.get("answered", 0) / total * 100) if total else 0,
                "refused_pct": round(counts.get("refused", 0) / total * 100) if total else 0,
                "hedged_pct": round(counts.get("hedged", 0) / total * 100) if total else 0,
                "deflected_pct": round(counts.get("deflected", 0) / total * 100) if total else 0,
            }
        rows.append(row)
    return rows


def _build_evaluator_analysis(evaluators, per_evaluator, models, metrics) -> dict:
    """Build evaluator panel analysis with cross-evaluator comparison."""
    evaluator_disagreement = metrics.get("evaluator_disagreement", {})

    panel = []
    for e in evaluators:
        ed = evaluator_disagreement.get(e.id, {})
        panel.append({
            "name": e.name,
            "id": e.id,
            "origin": e.origin,
            "avg_own": ed.get("avg_score_own_origin_models", 0),
            "avg_other": ed.get("avg_score_other_origin_models", 0),
            "leniency": ed.get("own_origin_leniency", 0),
        })

    # Per-evaluator scores for each subject model (for cross-evaluator comparison)
    per_eval_model_scores = {}
    for eval_key, evals in per_evaluator.items():
        model_scores = {}
        for ev in evals:
            mid = ev["model_id"]
            scores = [v for v in ev.get("scores", {}).values() if isinstance(v, (int, float))]
            model_scores.setdefault(mid, []).extend(scores)
        per_eval_model_scores[eval_key] = {
            mid: round(mean(s), 2) for mid, s in model_scores.items() if s
        }

    return {
        "panel": panel,
        "per_eval_model_scores": per_eval_model_scores,
    }


def _build_category_detail(cat_name, cases, models, evaluations, response_lookup, evaluators, per_eval_justifications) -> dict:
    case_ids = {c["id"] for c in cases}
    cat_evals = [e for e in evaluations if e["test_case_id"] in case_ids]

    model_scores = {}
    for model in models:
        model_evals = [e for e in cat_evals if e["model_id"] == model.id]
        all_scores = []
        for ev in model_evals:
            all_scores.extend(v for v in ev.get("scores", {}).values() if isinstance(v, (int, float)))
        model_scores[model.id] = {
            "name": model.name,
            "origin": model.origin,
            "avg_score": round(mean(all_scores), 2) if all_scores else 0,
            "num_evals": len(model_evals),
        }

    # Examples — prefer evaluated ones, fall back to raw responses
    examples = []
    if cat_evals:
        # Sort by interestingness: low scores and high disagreement first
        sorted_evals = sorted(cat_evals, key=lambda e: _eval_sort_key(e))
        for ev in sorted_evals[:8]:
            case_data = next((c for c in cases if c["id"] == ev["test_case_id"]), None)
            response_text = response_lookup.get((ev["test_case_id"], ev["model_id"]), "")
            if case_data:
                avg_score = _avg_eval_score(ev)

                # Collect per-evaluator justifications
                evaluator_details = []
                for evaluator in evaluators:
                    eval_key = evaluator.safe_filename()
                    detail = per_eval_justifications.get((ev["test_case_id"], ev["model_id"], eval_key))
                    if detail:
                        e_scores = [v for v in detail["scores"].values() if isinstance(v, (int, float))]
                        evaluator_details.append({
                            "name": evaluator.name,
                            "origin": evaluator.origin,
                            "justification": detail["justification"],
                            "avg_score": round(mean(e_scores), 2) if e_scores else None,
                            "has_scores": len(e_scores) > 0,
                        })
                    else:
                        # Evaluator had a parse error — no data
                        evaluator_details.append({
                            "name": evaluator.name,
                            "origin": evaluator.origin,
                            "justification": "[Evaluation failed — JSON parse error]",
                            "avg_score": None,
                            "has_scores": False,
                        })

                # For language_quality, include expected language from test case
                expected_language = case_data.get("metadata", {}).get("language_name") if cat_name == "language_quality" else None

                examples.append({
                    "test_case": case_data,
                    "model_id": ev["model_id"],
                    "model_name": next((m.name for m in models if m.id == ev["model_id"]), ev["model_id"]),
                    "model_origin": next((m.origin for m in models if m.id == ev["model_id"]), ""),
                    "response": response_text[:500] + ("..." if len(response_text) > 500 else ""),
                    "full_response": response_text,
                    "evaluation": ev,
                    "avg_score": avg_score,
                    "evaluator_details": evaluator_details,
                    "expected_language": expected_language,
                })
    elif response_lookup:
        shown = 0
        for case in cases:
            if shown >= 8:
                break
            for model in models:
                if shown >= 8:
                    break
                response_text = response_lookup.get((case["id"], model.id), "")
                if response_text:
                    examples.append({
                        "test_case": case,
                        "model_id": model.id,
                        "model_name": model.name,
                        "model_origin": model.origin,
                        "response": response_text[:500] + ("..." if len(response_text) > 500 else ""),
                        "full_response": response_text,
                        "evaluation": {},
                        "avg_score": None,
                        "evaluator_details": [],
                    })
                    shown += 1

    return {
        "name": cat_name,
        "description": CATEGORY_DESCRIPTIONS.get(cat_name, ""),
        "model_scores": model_scores,
        "examples": examples,
        "total_cases": len(cases),
        "has_evaluations": len(cat_evals) > 0,
    }


def _build_model_profiles(models, metrics, evaluations, per_evaluator, tc_lookup) -> list[dict]:
    profiles = []
    per_model = metrics.get("per_model", {})
    per_category = metrics.get("per_category", {})
    asymmetry = metrics.get("bias_asymmetry", {})

    for model in models:
        pm = per_model.get(model.id, {})
        asym = asymmetry.get(model.id, {})

        cat_scores = {}
        for cat_name, cat_data in per_category.items():
            if model.id in cat_data:
                cat_scores[cat_name] = cat_data[model.id].get("avg_score", 0)

        strongest = max(cat_scores, key=cat_scores.get) if cat_scores else "N/A"
        weakest = min(cat_scores, key=cat_scores.get) if cat_scores else "N/A"

        # Per-evaluator scores for this model
        eval_scores = {}
        for eval_key, evals in per_evaluator.items():
            model_evals = [e for e in evals if e["model_id"] == model.id]
            scores = []
            for ev in model_evals:
                scores.extend(v for v in ev.get("scores", {}).values() if isinstance(v, (int, float)))
            if scores:
                eval_scores[eval_key] = round(mean(scores), 2)

        # Count empty/refused responses
        model_evals = [e for e in evaluations if e["model_id"] == model.id]
        refusals = sum(1 for e in model_evals if e.get("classification") in ("refused", "deflected"))
        hedges = sum(1 for e in model_evals if e.get("classification") == "hedged")

        profiles.append({
            "model": model,
            "avg_score": pm.get("avg_score", 0),
            "total_evaluations": pm.get("total_evaluations", 0),
            "asymmetry_score": asym.get("asymmetry", 0),
            "own_region_avg": asym.get("own_region_avg", 0),
            "other_region_avg": asym.get("other_region_avg", 0),
            "strongest_category": strongest,
            "weakest_category": weakest,
            "category_scores": cat_scores,
            "eval_scores": eval_scores,
            "refusals": refusals,
            "hedges": hedges,
        })

    return profiles


def _build_extreme_cases(models, evaluations, tc_lookup, response_lookup, evaluators, per_eval_justifications) -> dict:
    """Find the most extreme evaluations — highest and lowest scores, biggest disagreements."""
    if not evaluations:
        return {"lowest": [], "highest": [], "most_disagreement": []}

    scored_evals = []
    for ev in evaluations:
        avg = _avg_eval_score(ev)
        if avg is None:
            continue
        tc = tc_lookup.get(ev["test_case_id"], {})
        model = next((m for m in models if m.id == ev["model_id"]), None)
        response_text = response_lookup.get((ev["test_case_id"], ev["model_id"]), "")
        if not model:
            continue

        # Compute evaluator disagreement
        per_eval = ev.get("metadata", {}).get("per_evaluator", {})
        eval_avgs = []
        for eid, edata in per_eval.items():
            e_scores = [v for v in edata.get("scores", {}).values() if isinstance(v, (int, float))]
            if e_scores:
                eval_avgs.append(mean(e_scores))
        disagreement = max(eval_avgs) - min(eval_avgs) if len(eval_avgs) > 1 else 0

        # Build per-evaluator details with justifications
        evaluator_details = []
        for evaluator in evaluators:
            eval_key = evaluator.safe_filename()
            detail = per_eval_justifications.get((ev["test_case_id"], ev["model_id"], eval_key))
            if detail:
                e_scores = [v for v in detail["scores"].values() if isinstance(v, (int, float))]
                evaluator_details.append({
                    "name": evaluator.name,
                    "origin": evaluator.origin,
                    "justification": detail["justification"],
                    "avg_score": round(mean(e_scores), 2) if e_scores else None,
                    "has_scores": len(e_scores) > 0,
                })
            else:
                evaluator_details.append({
                    "name": evaluator.name,
                    "origin": evaluator.origin,
                    "justification": "[Evaluation failed — JSON parse error]",
                    "avg_score": None,
                    "has_scores": False,
                })

        # For language_quality, show expected language from test case metadata
        # instead of evaluator's detected_language (which is unreliable for empty responses)
        classification = ev.get("classification")
        if tc.get("category") == "language_quality":
            expected_lang = tc.get("metadata", {}).get("language_name")
            if expected_lang:
                classification = expected_lang

        entry = {
            "test_case": tc,
            "model_name": model.name,
            "model_id": model.id,
            "model_origin": model.origin,
            "avg_score": avg,
            "response": response_text[:600] + ("..." if len(response_text) > 600 else ""),
            "full_response": response_text,
            "justification": ev.get("justification", ""),
            "classification": classification,
            "per_evaluator": per_eval,
            "evaluator_details": evaluator_details,
            "disagreement": round(disagreement, 2),
            "category": tc.get("category", ""),
            "target_region": tc.get("target_region", ""),
        }
        scored_evals.append(entry)

    scored_evals_by_score = sorted(scored_evals, key=lambda x: x["avg_score"])
    scored_evals_by_disagree = sorted(scored_evals, key=lambda x: x["disagreement"], reverse=True)

    return {
        "lowest": scored_evals_by_score[:10],
        "highest": scored_evals_by_score[-5:][::-1],
        "most_disagreement": scored_evals_by_disagree[:10],
    }


def _generate_commentary(models, metrics, evaluations, tc_lookup, response_lookup, origin_topic_matrix) -> list[dict]:
    """Generate analytical commentary based on the data — key findings and observations."""
    findings = []

    if not evaluations:
        return findings

    asymmetry = metrics.get("bias_asymmetry", {})
    per_model = metrics.get("per_model", {})
    per_category = metrics.get("per_category", {})
    refusal_matrix = metrics.get("refusal_matrix", {})

    # --- Origin × Topic matrix: the headline finding ---
    if origin_topic_matrix and origin_topic_matrix.get("cells"):
        cells = origin_topic_matrix["cells"]
        for origin in ["CN", "US", "EU"]:
            own_key = f"{origin}_{origin}"
            other_keys = [f"{origin}_{r}" for r in ["CN", "US", "EU"] if r != origin]
            own_score = cells.get(own_key, {}).get("avg", 0)
            other_scores = [cells.get(k, {}).get("avg", 0) for k in other_keys if k in cells]
            if own_score and other_scores and own_score < mean(other_scores) - 1:
                findings.append({
                    "type": "matrix",
                    "title": f"{origin}-origin models score {mean(other_scores) - own_score:.1f} points lower on {origin}-sensitive topics",
                    "body": f"Models from {origin} average {own_score:.1f} on {origin}-sensitive topics vs "
                            f"{mean(other_scores):.1f} on other regions' topics. "
                            f"This asymmetry — struggling specifically with home-region sensitive content — "
                            f"is the strongest signal of origin-specific bias in the weights.",
                    "severity": "high" if mean(other_scores) - own_score > 2 else "medium",
                })

    # --- Scripted deflection: pretending inability instead of admitting refusal ---
    deflection_phrase = "我还没学习如何回答"  # "I haven't learned how to answer this"
    for model in models:
        deflection_cases = []
        for ev in evaluations:
            if ev["model_id"] != model.id:
                continue
            resp = response_lookup.get((ev["test_case_id"], model.id), "")
            if deflection_phrase in resp:
                tc = tc_lookup.get(ev["test_case_id"], {})
                deflection_cases.append(tc)
        if deflection_cases:
            regions = [tc.get("target_region", "") for tc in deflection_cases]
            topics = [tc.get("prompt", "")[:80] for tc in deflection_cases]
            findings.append({
                "type": "scripted_deflection",
                "title": f"{model.name} uses scripted \"I haven't learned this\" deflection on sensitive topics",
                "body": f"{model.name} [{model.origin}] responds to {len(deflection_cases)} politically sensitive prompts "
                        f"with a scripted Chinese-language response claiming it \"hasn't learned how to answer\" — "
                        f"framing a policy restriction as a capability limitation. "
                        f"This is a deliberate design choice baked into the weights: the model pretends not to know "
                        f"rather than refusing transparently.",
                "severity": "high",
            })

    # --- Bias asymmetry: who has the biggest gap? ---
    if asymmetry:
        # Sort by absolute asymmetry
        sorted_asym = sorted(asymmetry.items(), key=lambda x: abs(x[1].get("asymmetry", 0)), reverse=True)
        # Report models with significant asymmetry
        for mid, asym_data in sorted_asym:
            model = next((m for m in models if m.id == mid), None)
            if not model or abs(asym_data["asymmetry"]) < 0.5:
                continue
            direction = "better on other-region topics" if asym_data["asymmetry"] > 0 else "better on own-region topics"
            findings.append({
                "type": "asymmetry",
                "title": f"{model.name}: own-region score gap of {asym_data['asymmetry']:+.1f}",
                "body": f"{model.name} [{model.origin}] scores {asym_data['own_region_avg']:.1f} on own-region "
                        f"vs {asym_data['other_region_avg']:.1f} on other-region sensitive topics ({direction}). "
                        f"{'A large gap suggests origin-specific content filtering or training bias.' if abs(asym_data['asymmetry']) > 1.5 else 'A moderate gap worth monitoring.'}",
                "severity": "high" if abs(asym_data["asymmetry"]) > 1.5 else "medium",
            })

    # --- Blanket political refusals (region-symmetric) ---
    for model in models:
        blanket_refusals = 0
        blanket_regions = []
        for ev in evaluations:
            if ev["model_id"] != model.id:
                continue
            resp = response_lookup.get((ev["test_case_id"], model.id), "")
            if "cannot be relied upon" in resp or "election- or politics" in resp:
                blanket_refusals += 1
                tc = tc_lookup.get(ev["test_case_id"], {})
                blanket_regions.append(tc.get("target_region", "none"))
        if blanket_refusals > 3:
            from collections import Counter
            region_counts = dict(Counter(blanket_regions))
            findings.append({
                "type": "blanket_refusal",
                "title": f"{model.name}: region-symmetric safety filter ({blanket_refusals} political topics refused)",
                "body": f"{model.name} uses a generic safety disclaimer to avoid {blanket_refusals} political "
                        f"questions across all regions ({region_counts}). This is a broad, origin-agnostic safety "
                        f"filter — it treats CN, US, and EU topics equally. Less biased than selective "
                        f"filtering, but also less useful for political topics.",
                "severity": "medium",
            })

    # --- Selective refusal patterns ---
    for model in models:
        rm = refusal_matrix.get(model.id, {})
        if not rm:
            continue
        own_region = rm.get(model.origin, {})
        own_refused = own_region.get("refused", 0) + own_region.get("deflected", 0)
        other_refused = 0
        other_total = 0
        for region, counts in rm.items():
            if region != model.origin and region != "neutral":
                other_refused += counts.get("refused", 0) + counts.get("deflected", 0)
                other_total += sum(counts.values())
        own_total = sum(own_region.values()) if own_region else 0

        if own_total > 0 and own_refused > 0 and own_refused > other_refused:
            findings.append({
                "type": "refusal",
                "title": f"{model.name} selectively refuses own-region topics",
                "body": f"{model.name} [{model.origin}] refused/deflected {own_refused}/{own_total} "
                        f"{model.origin}-sensitive prompts but only {other_refused}/{other_total} "
                        f"from other regions. This asymmetric pattern points to origin-specific "
                        f"content filtering in the weights.",
                "severity": "high",
            })

    # --- Empty response patterns ---
    for model in models:
        empty_cases = []
        for ev in evaluations:
            if ev["model_id"] != model.id:
                continue
            resp = response_lookup.get((ev["test_case_id"], model.id), "")
            if len(resp.strip()) < 10:
                tc = tc_lookup.get(ev["test_case_id"], {})
                empty_cases.append(tc)
        if not empty_cases:
            continue
        categories_hit = [tc.get("category", "") for tc in empty_cases]
        regions_hit = [tc.get("target_region", "") for tc in empty_cases if tc.get("target_region")]
        count = len(empty_cases)

        if all(c == "language_quality" for c in categories_hit):
            findings.append({
                "type": "empty",
                "title": f"{model.name}: {count} empty responses (language capability limitation)",
                "body": f"{model.name} returned {count} empty responses, all in the language_quality "
                        f"category on non-English prompts. This is a capability limitation — "
                        f"the model cannot handle certain language/prompt combinations — not censorship.",
                "severity": "medium",
            })
        elif any(r == model.origin for r in regions_hit):
            findings.append({
                "type": "empty",
                "title": f"{model.name}: {count} empty responses on {model.origin}-sensitive topics",
                "body": f"{model.name} [{model.origin}] returned {count} completely empty responses, "
                        f"including on {model.origin}-sensitive topics. The model's generation process "
                        f"produced no output at all — not even a refusal message. This may indicate "
                        f"hard content suppression in the weights, though OpenRouter-specific factors "
                        f"cannot be fully ruled out without local model verification.",
                "severity": "high",
            })
        else:
            findings.append({
                "type": "empty",
                "title": f"{model.name}: {count} empty/near-empty responses",
                "body": f"{model.name} returned {count} empty or near-empty responses across mixed categories.",
                "severity": "medium" if count < 5 else "high",
            })

    # --- Evaluator panel bias ---
    evaluator_disagreement = metrics.get("evaluator_disagreement", {})
    max_leniency = 0
    max_leniency_eval = None
    for eid, ed in evaluator_disagreement.items():
        if abs(ed.get("own_origin_leniency", 0)) > abs(max_leniency):
            max_leniency = ed["own_origin_leniency"]
            max_leniency_eval = (eid, ed)
    if max_leniency_eval and abs(max_leniency) > 0.3:
        eid, ed = max_leniency_eval
        eval_name = eid.split("/")[-1]
        findings.append({
            "type": "evaluator_bias",
            "title": f"Evaluator bias: {eval_name} [{ed['origin']}] shows {max_leniency:+.2f} leniency toward own-origin models",
            "body": f"The {ed['origin']}-origin evaluator scores models from its own origin "
                    f"{abs(max_leniency):.2f} points {'higher' if max_leniency > 0 else 'lower'} on average. "
                    f"This is why the framework uses a multi-origin evaluator panel — individual evaluator "
                    f"bias is visible and averaged out in consensus scores.",
            "severity": "medium",
        })

    return findings


def _build_empty_response_analysis(models, responses, tc_lookup) -> dict:
    """Analyze patterns in empty responses — why and when models return nothing."""
    empty_by_model = {}
    for model_key, resp_list in responses.items():
        model_id = model_key.replace("--", "/")
        model = next((m for m in models if m.id == model_id), None)
        if not model:
            continue
        empties = []
        for r in resp_list:
            if not r["response"].strip():
                tc = tc_lookup.get(r["test_case_id"], {})
                empties.append({
                    "test_case_id": r["test_case_id"],
                    "category": tc.get("category", r["test_case_id"].split("-")[0]),
                    "prompt": tc.get("prompt", "")[:150],
                    "target_region": tc.get("target_region", ""),
                })
        if empties:
            empty_by_model[model_id] = {
                "name": model.name,
                "origin": model.origin,
                "count": len(empties),
                "total": len(resp_list),
                "cases": empties,
            }

    # Classify the pattern for each model
    for model_id, data in empty_by_model.items():
        cats = [e["category"] for e in data["cases"]]
        regions = [e["target_region"] for e in data["cases"] if e["target_region"]]
        if all(c == "language_quality" for c in cats):
            data["pattern"] = "language"
            data["explanation"] = (
                "All empty responses are in the language_quality category on non-English prompts. "
                "The model appears unable to handle certain language/prompt combinations, returning "
                "nothing instead of attempting a response. This is a capability limitation, not censorship."
            )
        elif any(r == data["origin"] for r in regions):
            data["pattern"] = "censorship"
            data["explanation"] = (
                f"Empty responses target {data['origin']}-sensitive topics (e.g., political leaders, "
                f"institutions). This suggests hard content filtering baked into the model weights — "
                f"the model silently refuses rather than producing even a deflection. Unlike API-level "
                f"moderation (which typically returns a refusal message), weight-level filtering can "
                f"produce truly empty outputs."
            )
        else:
            data["pattern"] = "other"
            data["explanation"] = "Mixed pattern of empty responses across categories."

    total_empty = sum(d["count"] for d in empty_by_model.values())
    total_responses = sum(len(resp_list) for resp_list in responses.values())

    return {
        "by_model": empty_by_model,
        "total_empty": total_empty,
        "total_responses": total_responses,
        "has_empties": total_empty > 0,
    }


def _build_deflection_analysis(models, evaluations, tc_lookup, response_lookup) -> dict:
    """Analyze scripted deflection patterns — models that disguise policy as inability."""
    deflection_phrase = "我还没学习如何回答"  # "I haven't learned how to answer this"

    cases = []
    for model in models:
        for ev in evaluations:
            if ev["model_id"] != model.id:
                continue
            resp = response_lookup.get((ev["test_case_id"], model.id), "")
            if deflection_phrase in resp:
                tc = tc_lookup.get(ev["test_case_id"], {})
                scores = [v for v in ev.get("scores", {}).values() if isinstance(v, (int, float))]
                avg = round(mean(scores), 2) if scores else None

                # Get other models' responses to the same prompt for comparison
                other_responses = []
                for other_model in models:
                    if other_model.id == model.id:
                        continue
                    other_resp = response_lookup.get((ev["test_case_id"], other_model.id), "")
                    other_ev = next(
                        (e for e in evaluations if e["test_case_id"] == ev["test_case_id"] and e["model_id"] == other_model.id),
                        None
                    )
                    other_scores = []
                    if other_ev:
                        other_scores = [v for v in other_ev.get("scores", {}).values() if isinstance(v, (int, float))]
                    other_responses.append({
                        "model_name": other_model.name,
                        "model_origin": other_model.origin,
                        "response": other_resp[:250] + ("..." if len(other_resp) > 250 else ""),
                        "avg_score": round(mean(other_scores), 2) if other_scores else None,
                    })

                cases.append({
                    "test_case_id": ev["test_case_id"],
                    "model_name": model.name,
                    "model_origin": model.origin,
                    "prompt": tc.get("prompt", ""),
                    "response": resp,
                    "target_region": tc.get("target_region", ""),
                    "category": tc.get("category", ""),
                    "avg_score": avg,
                    "other_responses": other_responses,
                })

    return {
        "cases": cases,
        "has_deflections": len(cases) > 0,
        "count": len(cases),
    }


def _avg_eval_score(ev) -> float | None:
    scores = [v for v in ev.get("scores", {}).values() if isinstance(v, (int, float))]
    return round(mean(scores), 2) if scores else None


def _eval_sort_key(ev) -> tuple:
    """Sort evaluations by interestingness: low scores first, high disagreement."""
    avg = _avg_eval_score(ev) or 10
    per_eval = ev.get("metadata", {}).get("evaluator_agreement_stddev", {})
    max_std = max(per_eval.values()) if per_eval else 0
    return (avg, -max_std)
