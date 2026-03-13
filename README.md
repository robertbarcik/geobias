# GeoBias

**Systematic evaluation of geopolitical biases in open-weight language models.**

> *"If I deploy a Chinese-origin model on EU infrastructure, will it behave differently on politically sensitive topics than a US or European model? Is the bias in the weights, or just in the API moderation layer?"*

GeoBias answers this by running structured comparative evaluations across models from different geopolitical origins (US, EU, China) at comparable parameter sizes. It uses a multi-origin evaluator panel to design tests and assess results, producing an interactive HTML report showing exactly where biases appear, how severe they are, and whether they're symmetric or asymmetric.

**This is not about proving any model or origin is "bad."** US models may favor US narratives, EU models may reflect EU regulatory worldviews, Chinese models may be sensitive to Chinese state topics. GeoBias documents all of these asymmetries objectively. The value proposition: *here's the data, make an informed deployment decision.*

---

## How It Works

```
                         Subject Models
                    ┌──────────────────────┐
                    │  Qwen 8B      [CN]   │
                    │  Llama 8B     [US]   │
Identical ─────────>│  Phi-4        [US]   │────────> Responses
prompts             │  Ministral 8B [EU]   │
                    └──────────────────────┘
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
              ┌──────────┬──────────┬──────────┐
              │ Qwen 235B│ Llama 70B│ Mistral  │
              │   [CN]   │   [US]   │ Large[EU]│
              └────┬─────┴────┬─────┴────┬─────┘
                   │          │          │
                   ▼          ▼          ▼
              3 independent evaluations per response
                           │
                           ▼
                   Consensus scores
                + evaluator disagreement analysis
                           │
                           ▼
                   Interactive HTML report
```

### The Evaluator Bias Problem

Any single model used as a judge carries its own biases. A Chinese evaluator might go easy on China-sensitive topics; an American evaluator might normalize Western framing. GeoBias solves this with a **multi-origin evaluator panel** — three large models from different regions each judge every response independently:

| Evaluator | Origin | Cost (input/output per 1M tokens) |
|-----------|--------|-----------------------------------|
| Qwen3 235B | China | $0.07 / $0.10 |
| Llama 3.3 70B | US | $0.10 / $0.32 |
| Mistral Large 3 | EU | $0.50 / $1.50 |

Where all three agree, confidence is high. Where they disagree, the disagreement itself is informative data — and is surfaced in the report.

## Test Categories

| # | Category | What it tests |
|---|----------|---------------|
| 1 | **Narrative Framing** | How models frame geopolitical events — one-sided narratives, omitted perspectives, loaded language |
| 2 | **Refusal Asymmetry** | Do models refuse China-sensitive topics while freely discussing US-sensitive ones (or vice versa)? |
| 3 | **Entity Sentiment** | Sentiment bias when describing leaders (Xi, Biden, von der Leyen), institutions (CCP, NATO, EU), and countries |
| 4 | **Language Quality** | Performance across English, Chinese, German, French, Slovak — detecting training data bias |
| 5 | **Regulatory Values** | Do models favor their origin region's stance on data privacy, AI regulation, surveillance? |
| 6 | **Self-Awareness** | Can models honestly describe their own biases and limitations? |
| 7 | **Historical Narrative** | Control category — settled history (WWII, Cold War, colonialism) that all models should handle similarly |

For each category, the evaluator panel generates test prompts covering topics sensitive to each region (China, US, EU) plus neutral controls. All subject models receive identical prompts.

## Quick Start

```bash
git clone https://github.com/robertbarcik/geobias.git
cd geobias
python3 -m venv venv && source venv/bin/activate
pip install -e .

export OPENROUTER_API_KEY=your_key_here

# Preview what will run and estimated cost (no API calls)
python -m geobias evaluate --group 7b --dry-run

# Run the evaluation
python -m geobias evaluate --group 7b
```

## CLI

```bash
# Evaluate model groups
python -m geobias evaluate --group 7b           # ~7-14B models
python -m geobias evaluate --group 3b           # ~3-4B models
python -m geobias evaluate --group all          # Everything

# Pick specific models
python -m geobias evaluate --model qwen/qwen3-8b --model meta-llama/llama-3.1-8b-instruct

# Run specific categories
python -m geobias evaluate --group 7b --category refusal_asymmetry --category narrative_framing

# Budget mode: single evaluator
python -m geobias evaluate --group 7b --evaluator qwen/qwen3-235b-a22b-2507

# Control test depth
python -m geobias evaluate --group 7b --cases-per-category 12

# Regenerate report from existing data (no new API calls)
python -m geobias report --input ./reports/7b-run-20260313/

# List available models and categories
python -m geobias models
python -m geobias categories
```

## Pre-Configured Models

All model IDs verified against the OpenRouter API.

### ~3-4B Class

| Model | Origin | Params | ID |
|-------|--------|--------|----|
| Qwen3 4B | CN | 4B | `qwen/qwen3-4b:free` |
| Llama 3.2 3B | US | 3B | `meta-llama/llama-3.2-3b-instruct` |
| Gemma 3 4B | US | 4B | `google/gemma-3-4b-it` |
| Ministral 3B | EU | 3B | `mistralai/ministral-3b-2512` |

### ~7-14B Class

| Model | Origin | Params | ID |
|-------|--------|--------|----|
| Qwen3 8B | CN | 8B | `qwen/qwen3-8b` |
| Qwen3.5 9B | CN | 9B | `qwen/qwen3.5-9b` |
| Llama 3.1 8B | US | 8B | `meta-llama/llama-3.1-8b-instruct` |
| Phi-4 | US | 14B | `microsoft/phi-4` |
| Ministral 8B | EU | 8B | `mistralai/ministral-8b-2512` |

Adding models is straightforward — edit `DEFAULT_MODEL_GROUPS` in `src/geobias/config.py`.

## The Report

GeoBias produces a single self-contained HTML file (no external dependencies) with:

- **Executive summary** — key findings, bias rankings by asymmetry score
- **Refusal asymmetry heatmap** — the centerpiece: a matrix showing answer/hedge/deflect/refuse patterns per model per region
- **Evaluator panel analysis** — does each evaluator show leniency toward its own-origin models?
- **Per-category deep dives** — score comparisons, notable examples with full prompts and responses
- **Model profiles** — per-model summary with strongest/weakest categories
- **Methodology** — how tests were designed, scoring criteria, the exact evaluator system prompt (full transparency)
- **Raw data download** — one-click JSON export for further analysis

## Cost

The multi-origin evaluator panel is dramatically cheaper than using a frontier model as sole judge:

| Setup | 7B group (5 models, 7 categories, 8 cases each) | Notes |
|-------|--------------------------------------------------|-------|
| 3-evaluator panel (default) | ~$0.50 | Qwen 235B + Llama 70B + Mistral Large |
| Single budget evaluator | ~$0.15 | e.g., just Qwen 235B |
| Single Claude Sonnet (old approach) | ~$3.50 | And less rigorous |

Estimates are rough — actual cost depends on response lengths. Always use `--dry-run` first.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | *(required)* | Your OpenRouter API key |
| `GEOBIAS_EVALUATORS` | 3-model panel | Comma-separated evaluator model IDs |
| `GEOBIAS_CASES_PER_CATEGORY` | `8` | Number of test cases per category |

## Project Structure

```
src/geobias/
├── cli.py                  # Click CLI
├── config.py               # Model groups, evaluator panel, categories
├── llm.py                  # Async OpenRouter client with caching
├── pipeline.py             # Core evaluation pipeline
├── report.py               # HTML report generation
├── categories/
│   ├── base.py             # TestCase, Evaluation, TestCategory ABC
│   ├── prompts.py          # Shared evaluator system prompt, topic lists
│   ├── narrative_framing.py
│   ├── refusal_asymmetry.py
│   ├── entity_sentiment.py
│   ├── language_quality.py
│   ├── regulatory_values.py
│   ├── self_awareness.py
│   └── historical_narrative.py
└── templates/
    └── report.html         # Jinja2 template for the HTML report
```

## Extending

**Add a test category:** Create a new file in `src/geobias/categories/` subclassing `TestCategory`, register it in `registry.py`, and add the name to `ALL_CATEGORIES` in `config.py`.

**Add models:** Add a `ModelSpec` to the appropriate group in `config.py`. Use `python -m geobias models` to verify.

**Add a region:** Define new sensitive topics in `prompts.py` under `TOPIC_REGIONS`, then update the category prompts to include the new region.

## Limitations

- Results reflect behavior through the OpenRouter API, which may differ from native chat interfaces that apply additional moderation layers
- Evaluator models have their own biases — the multi-origin panel mitigates but doesn't eliminate this
- Test cases are AI-generated and may not cover all relevant scenarios
- Scores are relative, not absolute — most meaningful in comparison across models
- Model behavior changes with updates; results are point-in-time snapshots

## License

MIT
