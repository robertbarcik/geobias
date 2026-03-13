"""HTML report generation from evaluation results."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .categories.base import TestCase, Evaluation
from .categories.prompts import ORCHESTRATOR_SYSTEM_PROMPT
from .config import ModelSpec

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"


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

    return data


def generate_report(run_dir: Path) -> Path:
    """Generate an HTML report from run data. Returns path to the report file."""
    data = load_run_data(run_dir)

    if not data.get("config"):
        raise FileNotFoundError(f"No config.json found in {run_dir}")

    # Prepare template context
    models = [ModelSpec(**m) for m in data["config"]["models"]]
    categories = data["config"]["categories"]
    metrics = data.get("metrics", {})
    evaluations = data.get("evaluations", [])
    test_cases = data.get("test_cases", {})
    responses = data.get("responses", {})

    # Build response lookup: (test_case_id, model_safe_filename) -> response text
    response_lookup = {}
    for model_key, resp_list in responses.items():
        for r in resp_list:
            response_lookup[(r["test_case_id"], r["model_id"])] = r["response"]

    # Build evaluation lookup: (test_case_id, model_id) -> evaluation
    eval_lookup = {}
    for ev in evaluations:
        eval_lookup[(ev["test_case_id"], ev["model_id"])] = ev

    # Compute executive summary
    summary = _compute_executive_summary(models, metrics, evaluations, test_cases)

    # Build refusal heatmap data
    refusal_heatmap = _build_refusal_heatmap(models, metrics)

    # Build per-category data
    category_details = []
    for cat_name in categories:
        cat_cases = test_cases.get(cat_name, [])
        cat_data = _build_category_detail(cat_name, cat_cases, models, evaluations, response_lookup)
        category_details.append(cat_data)

    # Build model profiles
    model_profiles = _build_model_profiles(models, metrics, evaluations)

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html")

    # Build evaluator panel info
    evaluators = [ModelSpec(**e) for e in data["config"].get("evaluators", [])]
    evaluator_disagreement = metrics.get("evaluator_disagreement", {})

    html = template.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        run_dir=str(run_dir),
        models=models,
        categories=categories,
        evaluators=evaluators,
        evaluator_disagreement=evaluator_disagreement,
        summary=summary,
        refusal_heatmap=refusal_heatmap,
        category_details=category_details,
        model_profiles=model_profiles,
        metrics=metrics,
        orchestrator_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        raw_data=json.dumps(data, indent=2),
    )

    report_path = run_dir / "report.html"
    report_path.write_text(html)
    logger.info(f"Report generated: {report_path}")
    return report_path


def _compute_executive_summary(models, metrics, evaluations, test_cases) -> dict:
    """Compute high-level summary stats."""
    per_model = metrics.get("per_model", {})
    asymmetry = metrics.get("bias_asymmetry", {})

    model_summaries = []
    for model in models:
        pm = per_model.get(model.id, {})
        asym = asymmetry.get(model.id, {})
        model_summaries.append({
            "name": model.name,
            "origin": model.origin,
            "avg_score": pm.get("avg_score", 0),
            "asymmetry_score": asym.get("asymmetry", 0),
            "total_evaluations": pm.get("total_evaluations", 0),
        })

    # Overall bias ranking
    model_summaries.sort(key=lambda x: abs(x["asymmetry_score"]), reverse=True)

    total_test_cases = sum(len(cases) for cases in test_cases.values())
    total_evaluations = len(evaluations)

    return {
        "total_models": len(models),
        "total_test_cases": total_test_cases,
        "total_evaluations": total_evaluations,
        "model_summaries": model_summaries,
    }


def _build_refusal_heatmap(models, metrics) -> list[dict]:
    """Build data for the refusal asymmetry heatmap."""
    refusal_matrix = metrics.get("refusal_matrix", {})
    rows = []
    for model in models:
        matrix = refusal_matrix.get(model.id, {})
        row = {
            "model_name": model.name,
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


def _build_category_detail(cat_name, cases, models, evaluations, response_lookup) -> dict:
    """Build detailed data for a single category section."""
    case_ids = {c["id"] for c in cases}

    # Get evaluations for this category
    cat_evals = [e for e in evaluations if e["test_case_id"] in case_ids]

    # Per-model scores
    model_scores = {}
    for model in models:
        model_evals = [e for e in cat_evals if e["model_id"] == model.id]
        all_scores = []
        for ev in model_evals:
            all_scores.extend(v for v in ev.get("scores", {}).values() if isinstance(v, (int, float)))
        model_scores[model.id] = {
            "name": model.name,
            "origin": model.origin,
            "avg_score": round(sum(all_scores) / len(all_scores), 2) if all_scores else 0,
            "num_evals": len(model_evals),
        }

    # Notable examples (up to 5)
    examples = []
    for ev in cat_evals[:5]:
        case_data = next((c for c in cases if c["id"] == ev["test_case_id"]), None)
        response_text = response_lookup.get((ev["test_case_id"], ev["model_id"]), "")
        if case_data:
            examples.append({
                "test_case": case_data,
                "model_id": ev["model_id"],
                "model_name": next((m.name for m in models if m.id == ev["model_id"]), ev["model_id"]),
                "response": response_text[:500] + ("..." if len(response_text) > 500 else ""),
                "full_response": response_text,
                "evaluation": ev,
            })

    CATEGORY_DESCRIPTIONS = {
        "narrative_framing": "How models frame geopolitical events — detecting one-sided narratives and loaded language.",
        "refusal_asymmetry": "Patterns of refusals and deflections across topics sensitive to different regions.",
        "entity_sentiment": "Sentiment bias in descriptions of political leaders, institutions, and countries.",
        "language_quality": "Language quality across multiple languages, detecting training data bias.",
        "regulatory_values": "Whether models favor their origin region's regulatory stance.",
        "self_awareness": "Model self-awareness about biases, limitations, and training.",
        "historical_narrative": "Control category — historically settled topics that should be handled similarly by all models.",
    }

    return {
        "name": cat_name,
        "description": CATEGORY_DESCRIPTIONS.get(cat_name, ""),
        "model_scores": model_scores,
        "examples": examples,
        "total_cases": len(cases),
    }


def _build_model_profiles(models, metrics, evaluations) -> list[dict]:
    """Build per-model profile summaries."""
    profiles = []
    per_model = metrics.get("per_model", {})
    per_category = metrics.get("per_category", {})
    asymmetry = metrics.get("bias_asymmetry", {})

    for model in models:
        pm = per_model.get(model.id, {})
        asym = asymmetry.get(model.id, {})

        # Find strongest/weakest categories
        cat_scores = {}
        for cat_name, cat_data in per_category.items():
            if model.id in cat_data:
                cat_scores[cat_name] = cat_data[model.id].get("avg_score", 0)

        strongest = max(cat_scores, key=cat_scores.get) if cat_scores else "N/A"
        weakest = min(cat_scores, key=cat_scores.get) if cat_scores else "N/A"

        # Notable behaviors
        model_evals = [e for e in evaluations if e["model_id"] == model.id]
        notable = []
        for ev in model_evals:
            if ev.get("highlights"):
                for h in ev["highlights"][:2]:
                    notable.append(h)
            if len(notable) >= 5:
                break

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
            "notable_behaviors": notable,
        })

    return profiles
