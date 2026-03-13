# GeoBias

Systematic evaluation of geopolitical biases in small open-weight language models.

**Core question:** "If I deploy a Chinese-origin model on EU infrastructure, will it behave differently on politically sensitive topics than a US or European model? Is the bias in the weights, or just in the API moderation layer?"

This is a balanced, comparative analysis — US, EU, and Chinese models are all evaluated by the same criteria. The output is an interactive HTML report showing where biases appear, how severe they are, and whether they're symmetric or asymmetric.

## Quick Start

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -e .
export OPENROUTER_API_KEY=your_key_here

# Dry run (cost estimate, no API calls)
python -m geobias evaluate --group 7b --dry-run

# Run evaluation
python -m geobias evaluate --group 7b

# Regenerate report from existing data
python -m geobias report --input ./reports/7b-run-20260313-143000/
```

## CLI Reference

```bash
python -m geobias evaluate --group 7b                     # All 7B-class models, default evaluator panel
python -m geobias evaluate --group 3b                     # All 3B-class models
python -m geobias evaluate --group all                    # All model groups
python -m geobias evaluate --model qwen/qwen3-8b --model meta-llama/llama-3.1-8b-instruct
python -m geobias evaluate --group 7b --category refusal_asymmetry --category narrative_framing
python -m geobias evaluate --group 7b --cases-per-category 10
python -m geobias evaluate --group 7b --evaluator qwen/qwen3-235b-a22b-2507  # Single evaluator (budget)
python -m geobias evaluate --group 7b --output ./my-reports
python -m geobias models                                   # List model groups + evaluator panel
python -m geobias categories                               # List test categories
python -m geobias report --input ./reports/run-dir/        # Regenerate report
```

## Environment Variables

```
OPENROUTER_API_KEY=...                          # Required
GEOBIAS_EVALUATORS=qwen/...,meta-llama/...,... # Comma-separated evaluator IDs (default: 3-model panel)
GEOBIAS_CASES_PER_CATEGORY=8                    # Test cases per category (default: 8)
```

## Architecture

```
Evaluator Panel (CN: Qwen 235B + US: Llama 70B + EU: Mistral Large)
  │
  ├── 1. Generate test prompts (first evaluator generates)
  │     └── Same prompts sent to ALL subject models
  │
  ├── 2. Collect responses (subject models respond)
  │     └── Each model gets identical prompts
  │
  ├── 3. Each evaluator independently judges every response
  │     └── 3 independent scores + justifications per response
  │
  ├── 3b. Merge into consensus scores (average + agreement stats)
  │     └── Where evaluators disagree → data about evaluator bias
  │
  └── 4. Compute metrics + generate HTML report
```

### Key design decisions:
- **Multi-origin evaluator panel**: Instead of a single judge (which would have its own biases), 3 evaluators from CN/US/EU origins each judge independently. Consensus scores average across them; disagreement patterns reveal evaluator bias. This is both cheaper than Claude Sonnet and more methodologically sound.
- **Evaluator bias detection**: The report includes an "Evaluator Panel" section showing whether each evaluator is more lenient toward models from its own origin. This meta-analysis makes the whole framework more credible.
- **Prompt consistency**: All models receive identical prompts per test case. No per-model variation.
- **Response caching**: All API responses cached by content hash. Re-runs skip already-fetched responses.
- **Resumability**: If interrupted, re-running picks up where it left off (checks for existing responses/evaluations).

## Model Configuration

Models defined in `src/geobias/config.py` → `DEFAULT_MODEL_GROUPS`. Each model has:
- `id`: OpenRouter model ID
- `name`: Human-readable name
- `origin`: Region code ("CN", "US", "EU")
- `params_b`: Parameter count in billions
- `family`: Model family name

To add a model, add a `ModelSpec` to the appropriate group in `DEFAULT_MODEL_GROUPS`. To add a new size group, add a new `ModelGroup` entry.

## Test Categories

All categories in `src/geobias/categories/`. Each is a `TestCategory` subclass with:
- `generation_prompt()` — creates the orchestrator prompt to generate test cases
- `parse_test_cases()` — parses orchestrator output into `TestCase` objects
- `evaluation_prompt()` — creates the orchestrator prompt to evaluate a model response
- `parse_evaluation()` — parses evaluation output into `Evaluation` objects

### Categories:
1. **narrative_framing** — Geopolitical event framing bias
2. **refusal_asymmetry** — Refusal/deflection patterns across regions (the "DeepSeek effect")
3. **entity_sentiment** — Sentiment in descriptions of leaders/institutions/countries
4. **language_quality** — Quality across languages (EN, ZH, DE, FR, SK)
5. **regulatory_values** — Regulatory stance bias (GDPR, AI regulation, etc.)
6. **self_awareness** — Model transparency about its own biases
7. **historical_narrative** — Control category (historically settled topics)

### Adding a new category:
1. Create `src/geobias/categories/my_category.py` subclassing `TestCategory`
2. Register it in `src/geobias/categories/registry.py`
3. Add the name to `ALL_CATEGORIES` in `src/geobias/config.py`

## Orchestrator Prompts

The orchestrator system prompt (in `src/geobias/categories/prompts.py`) is critical for balanced evaluation. It explicitly instructs:
- Evaluate all models by the same criteria
- Do NOT assume any model is the "default" neutral baseline
- Flag bias in ALL directions equally
- Base evaluation solely on response content, not assumptions about origin

This prompt is included in the HTML report's Methodology section for transparency.

## Report

Single self-contained HTML file with inline CSS/JS. Dark theme, interactive:
- Executive summary with bias rankings
- Refusal asymmetry heatmap (the centerpiece)
- Per-category deep dives with score charts and examples
- Model profiles with per-category breakdowns
- Methodology section (includes orchestrator prompt)
- Raw data download button

## Data Persistence

Results are stored in a **separate repo**: [geobias-results](https://github.com/robertbarcik/geobias-results). When `../geobias-results/` exists as a sibling directory, GeoBias writes there automatically. Otherwise falls back to `./reports/`. Override with `--output`.

```
../geobias-results/<group>-run-<timestamp>/
├── config.json                    # Run configuration
├── test_cases.json                # Generated test prompts
├── responses/                     # Per-model response files
│   ├── qwen--qwen3-8b.json
│   └── ...
├── evaluations.json               # Consensus evaluations (merged across evaluators)
├── evaluations_per_evaluator/     # Raw per-evaluator judgments
├── metrics.json                   # Computed comparative metrics
├── report.html                    # Interactive report
└── .cache/                        # LLM response cache (gitignored)
```

## Ethical Considerations

- This tool documents biases objectively in ALL directions. It is not designed to prove any particular model or origin is "bad."
- Results reflect behavior through OpenRouter's API, which may differ from native interfaces with additional moderation.
- The orchestrator model (Claude) has its own biases — this is acknowledged in the report methodology.
- Results are point-in-time snapshots; model behavior changes with updates.
- Test cases are AI-generated and may not cover all relevant scenarios.
