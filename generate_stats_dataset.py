"""Generates a small, stats/data-science-specific instruction fine-tuning
dataset via the local qwen3-coder:30b model (free, no API cost). Topics are
chosen to match the target domain: applied statistics, fraud/risk models,
time-series forecasting, count regression -- the areas a general dataset
like MathInstruct doesn't cover.

This is a SEED dataset for review, not a finished training set: read
stats_dataset.jsonl afterward, spot-check a sample of each topic for
correctness before training on it. LLM-generated stats content can contain
subtly wrong explanations, which is worse than no data if trained on as-is.
"""
import json
import re
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3-coder:30b"
PER_TOPIC = 20
OUTPUT_PATH = "stats_dataset.jsonl"

TOPICS = [
    "descriptive statistics and summary measures (mean, median, variance, skewness)",
    "probability distributions (binomial, Poisson, normal, exponential) and when to use each",
    "hypothesis testing, p-values, and confidence intervals",
    "ANOVA and ANCOVA for comparing group means",
    "linear regression, coefficients, and residual diagnostics",
    "logistic regression and classification metrics (precision, recall, F1)",
    "Poisson and negative binomial regression for count data, including overdispersion",
    "time series basics: stationarity, autocorrelation, differencing",
    "ARIMA/SARIMA forecasting model selection and interpretation",
    "fraud detection and imbalanced classification (ROC-AUC, class imbalance handling)",
    "risk scoring and model evaluation in a business context",
    "experimental design basics (randomization, control groups, confounding)",
]

PROMPT_TEMPLATE = """Generate exactly {n} diverse instruction-tuning examples about: {topic}

Requirements:
- Mix of difficulty: some basic definitional questions, some applied/practical scenarios, some "explain the difference between X and Y" questions.
- Each "output" must be technically correct, concise (2-5 sentences), and written like an experienced statistician explaining to a colleague -- no fluff.
- Vary phrasing and question style across examples -- do not repeat similar wording.
- Return ONLY a valid JSON array, no markdown fences, no commentary. Format:
[{{"instruction": "...", "output": "..."}}, ...]
"""


def _extract_json_array(text: str) -> list:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("No JSON array found in model output")
    return json.loads(text[start:end + 1])


def generate_for_topic(topic: str, n: int) -> list:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(n=n, topic=topic)}],
        "stream": False,
        "options": {"temperature": 0.8},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    content = body["message"]["content"]
    return _extract_json_array(content)


def main():
    all_rows = []
    for topic in TOPICS:
        print(f"Generating {PER_TOPIC} examples for: {topic}")
        try:
            rows = generate_for_topic(topic, PER_TOPIC)
        except Exception as e:
            print(f"  FAILED ({e}), skipping topic")
            continue
        valid = [r for r in rows if isinstance(r, dict) and r.get("instruction") and r.get("output")]
        print(f"  got {len(valid)}/{len(rows) if isinstance(rows, list) else '?'} valid rows")
        for r in valid:
            r["topic"] = topic
        all_rows.extend(valid)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
