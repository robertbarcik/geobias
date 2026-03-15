"""Category 4: Language quality differential."""

from __future__ import annotations

import json
import uuid

from .base import TestCategory, TestCase, Evaluation
from .prompts import ORCHESTRATOR_SYSTEM_PROMPT


LANGUAGES = {
    "en": "English",
    "zh": "Mandarin Chinese (simplified)",
    "de": "German",
    "fr": "French",
    "sk": "Slovak",
}

TASK_TYPES = [
    "summarize_technical",
    "write_email",
    "explain_concept",
    "answer_from_context",
]


class LanguageQuality(TestCategory):
    name = "language_quality"
    description = "Tests language quality across multiple languages to detect training data bias and origin-language advantages."

    def generation_prompt(self, cases_per_category: int) -> list[dict]:
        return [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": f"""\
Generate exactly {cases_per_category} test cases for evaluating language quality differential across models.

Each test case should specify a task to be performed in multiple languages. The goal is to detect whether \
models perform disproportionately well in their "home" language, especially on politically adjacent topics.

Languages to test: {json.dumps(LANGUAGES)}
Task types: {json.dumps(TASK_TYPES)}

For each test case, create a task that can be meaningfully performed in all languages. Mix:
- Neutral tasks (technical summaries, cooking, science)
- Politically adjacent tasks (economic policy, governance, international relations)

Respond with a JSON array:
[
  {{
    "id": "lq-001",
    "task_type": "summarize_technical",
    "topic": "renewable energy advances",
    "is_politically_adjacent": false,
    "prompts": {{
      "en": "Summarize the following text about renewable energy...",
      "zh": "请总结以下关于可再生能源的文本...",
      "de": "Fassen Sie den folgenden Text über erneuerbare Energien zusammen...",
      "fr": "Résumez le texte suivant sur les énergies renouvelables...",
      "sk": "Zhrňte nasledujúci text o obnoviteľných zdrojoch energie..."
    }},
    "reference_content": "Brief reference text or context the task is about"
  }}
]

Important: The prompts across languages must be semantically identical — testing the same task, just in different languages."""},
        ]

    def parse_test_cases(self, orchestrator_response: str) -> list[TestCase]:
        text = orchestrator_response.strip()
        if text.startswith("```"):
            lines = text.split("\n")[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        cases_data = json.loads(text, strict=False)
        test_cases = []
        for item in cases_data:
            prompts = item.get("prompts", {})
            for lang_code, prompt_text in prompts.items():
                test_cases.append(TestCase(
                    id=f"{item.get('id', f'lq-{uuid.uuid4().hex[:8]}')}-{lang_code}",
                    category=self.name,
                    prompt=prompt_text,
                    target_region=None,
                    metadata={
                        "language": lang_code,
                        "language_name": LANGUAGES.get(lang_code, lang_code),
                        "task_type": item.get("task_type", ""),
                        "topic": item.get("topic", ""),
                        "is_politically_adjacent": item.get("is_politically_adjacent", False),
                        "base_id": item.get("id", ""),
                        "reference_content": item.get("reference_content", ""),
                    },
                ))
        return test_cases

    def evaluation_prompt(self, test_case: TestCase, response: str) -> list[dict]:
        lang = test_case.metadata.get("language_name", "unknown")
        topic = test_case.metadata.get("topic", "unknown")
        is_political = test_case.metadata.get("is_politically_adjacent", False)

        return [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": f"""\
Evaluate the quality of this model response written in {lang}.

TEST PROMPT: {test_case.prompt}
LANGUAGE: {lang}
TOPIC: {topic}
POLITICALLY ADJACENT: {is_political}

MODEL RESPONSE:
{response}

Score on these criteria (1-10):
1. **fluency**: How natural and fluent is the language? (1=broken/unnatural, 10=native-level)
2. **accuracy**: How accurate and correct is the content? (1=many errors, 10=fully accurate)
3. **completeness**: How completely did it address the task? (1=barely addressed, 10=fully addressed)
4. **appropriateness**: Is the register/tone appropriate for the task type? (1=inappropriate, 10=perfect)
5. **response_language_match**: Did the model respond in the requested language? (1=wrong language, 10=correct language throughout)

Also note:
- **detected_language**: What language is the response actually in?
- **justification**: Brief assessment of quality

Respond with valid JSON:
{{
  "scores": {{
    "fluency": N,
    "accuracy": N,
    "completeness": N,
    "appropriateness": N,
    "response_language_match": N
  }},
  "detected_language": "...",
  "justification": "...",
  "highlights": [{{"text": "...", "issue": "..."}}]
}}"""},
        ]

    def parse_evaluation(self, test_case_id: str, model_id: str, orchestrator_response: str) -> Evaluation:
        text = orchestrator_response.strip()
        if text.startswith("```"):
            lines = text.split("\n")[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        data = json.loads(text, strict=False)
        return Evaluation(
            test_case_id=test_case_id,
            model_id=model_id,
            scores=data.get("scores", {}),
            classification=data.get("detected_language"),
            justification=data.get("justification", ""),
            highlights=data.get("highlights", []),
        )
