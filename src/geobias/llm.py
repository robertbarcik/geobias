"""LLM client for OpenRouter API calls with caching and rate limiting."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict
    cached: bool = False


class LLMClient:
    """Async OpenRouter client with response caching and rate limiting."""

    def __init__(self, cache_dir: Optional[Path] = None, requests_per_minute: int = 200):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY environment variable is required")

        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.cache_dir = cache_dir
        self.rpm_limit = requests_per_minute
        self._semaphore = asyncio.Semaphore(20)  # max concurrent requests
        self._last_request_time = 0.0
        self._min_interval = 60.0 / requests_per_minute

    def _cache_key(self, model: str, messages: list[dict], temperature: float) -> str:
        payload = json.dumps({"model": model, "messages": messages, "temperature": temperature}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[LLMResponse]:
        if not self.cache_dir:
            return None
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            data = json.loads(cache_file.read_text())
            return LLMResponse(
                content=data["content"],
                model=data["model"],
                usage=data.get("usage", {}),
                cached=True,
            )
        return None

    def _save_cache(self, key: str, response: LLMResponse) -> None:
        if not self.cache_dir:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"{key}.json"
        cache_file.write_text(json.dumps({
            "content": response.content,
            "model": response.model,
            "usage": response.usage,
        }))

    async def complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a chat completion request with caching and rate limiting."""
        cache_key = self._cache_key(model, messages, temperature)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        async with self._semaphore:
            # Rate limiting
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_request_time = time.monotonic()

            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        content = response.choices[0].message.content or ""
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        result = LLMResponse(content=content, model=model, usage=usage)
        self._save_cache(cache_key, result)
        return result

    @staticmethod
    def _strip_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return text

    async def complete_json(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> dict:
        """Complete and parse the response as JSON.

        One malformed generation should not kill a whole run: on a parse
        failure we retry once, telling the model exactly what was wrong and
        asking for corrected JSON only. A second failure raises with enough
        context to diagnose (model + the start of the offending text).
        """
        resp = await self.complete(model, messages, temperature, max_tokens)
        text = self._strip_fences(resp.content)
        try:
            return json.loads(text, strict=False)
        except json.JSONDecodeError as first_err:
            retry_messages = messages + [
                {"role": "assistant", "content": resp.content},
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was not valid JSON "
                        f"(parser error: {first_err}). Send the corrected, "
                        "complete JSON only, with no commentary and no "
                        "markdown fences."
                    ),
                },
            ]
            resp2 = await self.complete(model, retry_messages, temperature, max_tokens)
            text2 = self._strip_fences(resp2.content)
            try:
                return json.loads(text2, strict=False)
            except json.JSONDecodeError as second_err:
                raise RuntimeError(
                    f"Model {model} returned invalid JSON twice "
                    f"(last error: {second_err}). Response started with: "
                    f"{text2[:200]!r}"
                ) from second_err


def estimate_cost(
    num_models: int,
    categories: list[str],
    cases_per_category: int,
    num_evaluators: int = 3,
) -> dict:
    """Rough cost estimate for an evaluation run.

    Uses the real expanded case count per category (language_quality expands
    each generated case into one TestCase per language, a 5x multiplier).
    """
    from .categories.registry import get_categories

    cats = get_categories(list(categories))
    total_cases = sum(cat.expected_case_count(cases_per_category) for cat in cats)

    generation_calls = len(cats)  # one batch generation per category
    response_calls = num_models * total_cases
    evaluation_calls = response_calls * num_evaluators  # each evaluator judges every response

    # Token estimates measured from real 7b runs: subject prompts are short
    # (~50 tokens) but responses average ~1100 completion tokens. Evaluation
    # prompts embed the full subject response plus the scoring rubric.
    subject_prompt_tokens = 100
    subject_completion_tokens = 1100
    eval_prompt_tokens = 500 + subject_completion_tokens
    eval_completion_tokens = 500

    # Pricing: small subject models ~$0.10/M in and out; evaluator panel avg
    # (Qwen 235B: $0.07/$0.10, Llama 70B: $0.10/$0.32, Mistral Large: $0.50/$1.50)
    # is roughly $0.22/M input, $0.64/M output.
    small_model_cost = response_calls * (
        subject_prompt_tokens * 0.10 + subject_completion_tokens * 0.10
    ) / 1_000_000
    evaluator_cost = (
        generation_calls * (500 * 0.22 + 2000 * 0.64)  # batch generations are long JSON outputs
        + evaluation_calls * (eval_prompt_tokens * 0.22 + eval_completion_tokens * 0.64)
    ) / 1_000_000

    return {
        "generation_calls": generation_calls,
        "response_calls": response_calls,
        "evaluation_calls": evaluation_calls,
        "total_api_calls": generation_calls + response_calls + evaluation_calls,
        "estimated_cost_usd": round(small_model_cost + evaluator_cost, 2),
        "note": "Rough estimate. Actual cost depends on model pricing and response lengths.",
    }
