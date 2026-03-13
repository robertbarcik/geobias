"""CLI interface for GeoBias."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click

from .config import (
    ALL_CATEGORIES,
    DEFAULT_EVALUATORS,
    DEFAULT_MODEL_GROUPS,
    EvalConfig,
    ModelSpec,
    get_cases_per_category,
    get_evaluators,
    resolve_models,
)
from .llm import estimate_cost


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.version_option(package_name="geobias")
def cli():
    """GeoBias — Systematic evaluation of geopolitical biases in language models."""
    pass


@cli.command()
@click.option("--group", type=str, default=None, help="Model group to evaluate (3b, 7b, or all)")
@click.option("--model", "model_ids", type=str, multiple=True, help="Specific model ID(s) to evaluate")
@click.option("--category", "categories", type=str, multiple=True, help="Test categories to run")
@click.option("--cases-per-category", type=int, default=None, help="Number of test cases per category")
@click.option("--evaluator", "evaluator_ids", type=str, multiple=True, help="Evaluator model ID(s). Default: multi-origin panel (CN+US+EU)")
@click.option("--output", type=click.Path(), default="reports", help="Output directory")
@click.option("--dry-run", is_flag=True, help="Show plan and cost estimate without making API calls")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def evaluate(group, model_ids, categories, cases_per_category, evaluator_ids, output, dry_run, verbose):
    """Run a geopolitical bias evaluation."""
    setup_logging(verbose)

    # Resolve models
    if not group and not model_ids:
        click.echo("Error: specify --group or --model", err=True)
        sys.exit(1)

    try:
        models = resolve_models(group=group, model_ids=list(model_ids) if model_ids else None)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Resolve evaluators
    if evaluator_ids:
        evaluators = []
        known = {e.id: e for e in DEFAULT_EVALUATORS}
        for eid in evaluator_ids:
            if eid in known:
                evaluators.append(known[eid])
            else:
                evaluators.append(ModelSpec(eid, eid.split("/")[-1], "unknown", 0, "unknown"))
    else:
        evaluators = get_evaluators()

    # Resolve categories
    cats = list(categories) if categories else ALL_CATEGORIES

    # Resolve cases per category
    cpc = cases_per_category or get_cases_per_category()

    config = EvalConfig(
        models=models,
        categories=cats,
        cases_per_category=cpc,
        evaluators=evaluators,
        output_dir=Path(output),
        group_name=group,
        dry_run=dry_run,
    )

    # Show plan
    click.echo(f"\n{'='*60}")
    click.echo("GeoBias Evaluation Plan")
    click.echo(f"{'='*60}")
    click.echo(f"\nEvaluator panel ({len(evaluators)}):")
    for e in evaluators:
        click.echo(f"  {e.name:30s} [{e.origin}]  ({e.id})")
    click.echo(f"\nCategories:   {', '.join(cats)}")
    click.echo(f"Cases/cat:    {cpc}")
    click.echo(f"\nSubject models ({len(models)}):")
    for m in models:
        click.echo(f"  {m.name:30s} [{m.origin}] {m.params_b}B  ({m.id})")

    # Cost estimate
    cost = estimate_cost(len(models), len(cats), cpc, len(evaluators))
    click.echo(f"\nEstimated API calls: {cost['total_api_calls']}")
    click.echo(f"  - Test generation:  {cost['generation_calls']}")
    click.echo(f"  - Model responses:  {cost['response_calls']}")
    click.echo(f"  - Evaluations:      {cost['evaluation_calls']} ({len(evaluators)} evaluators × {cost['evaluation_calls'] // len(evaluators)} each)")
    click.echo(f"Estimated cost:      ~${cost['estimated_cost_usd']:.2f} ({cost['note']})")

    if dry_run:
        click.echo("\n[DRY RUN] No API calls made.")
        return

    click.echo(f"\n{'='*60}")
    click.echo("Starting evaluation...")
    click.echo(f"{'='*60}\n")

    from .pipeline import EvaluationPipeline

    pipeline = EvaluationPipeline(config)
    run_dir = asyncio.run(pipeline.run())

    # Generate report
    click.echo("\nGenerating HTML report...")
    from .report import generate_report

    report_path = generate_report(run_dir)

    click.echo(f"\nDone! Results saved to: {run_dir}")
    click.echo(f"Report: {report_path}")


@cli.command()
@click.option("--input", "input_dir", type=click.Path(exists=True), required=True, help="Path to run directory")
def report(input_dir):
    """Regenerate HTML report from existing run data."""
    setup_logging(False)

    from .report import generate_report

    run_dir = Path(input_dir)
    report_path = generate_report(run_dir)
    click.echo(f"Report generated: {report_path}")


@cli.command()
def models():
    """List available model groups and models."""
    click.echo("\nSubject model groups:")
    for group_name, group in DEFAULT_MODEL_GROUPS.items():
        click.echo(f"\n  {group_name}: {group.description}")
        for m in group.models:
            click.echo(f"    {m.name:30s} [{m.origin}] {m.params_b}B  ({m.id})")

    click.echo("\nDefault evaluator panel:")
    for e in DEFAULT_EVALUATORS:
        click.echo(f"    {e.name:30s} [{e.origin}] {e.params_b}B  ({e.id})")


@cli.command()
def categories():
    """List available test categories."""
    from .categories.registry import CATEGORY_REGISTRY

    for name, cat in CATEGORY_REGISTRY.items():
        click.echo(f"  {name:25s} {cat.description}")
