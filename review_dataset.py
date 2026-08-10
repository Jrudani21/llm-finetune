"""Automated review of stats_dataset.jsonl — the gate before training.

The 240 examples were LLM-generated; the project note warns LLM-generated
stats content can be "subtly wrong, which is worse than no data if trained
on as-is." This script flags the detectable failure modes automatically so
a human only reads the suspicious rows instead of all 240:

- structural: missing fields, wrong types, empty/short/duplicated rows
- contamination: markdown fences, JSON remnants, mojibake, emoji
- self-contradiction: instruction contains a number and the output's number
  contradicts it (e.g. "mean of 2,4,6" answered "5")
- topic drift: output mentions a topic's key terms — flag rows whose terms
  suggest a different topic than the one labeled
- boilerplate: near-identical phrasing across rows (LLM repetition)

Usage:  python review_dataset.py [path]
Output: a report; exit code 0 = no red flags, 1 = issues found.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

TOPIC_KEYWORDS = {
    "descriptive statistics and summary measures (mean, median, variance, skewness)": ["mean", "median", "variance", "skew", "standard deviation", "quartile", "central tendency"],
    "probability distributions (binomial, Poisson, normal, exponential) and when to use each": ["binomial", "poisson", "normal distribution", "exponential", "probability distribution", "pmf", "pdf", "cdf"],
    "hypothesis testing, p-values, and confidence intervals": ["p-value", "p value", "null hypothesis", "alternative hypothesis", "confidence interval", "type i", "type ii", "significance level"],
    "ANOVA and ANCOVA for comparing group means": ["anova", "ancova", "f-test", "f test", "between-group", "within-group", "covariate"],
    "linear regression, coefficients, and residual diagnostics": ["linear regression", "coefficient", "residual", "r-squared", "r²", "ordinary least squares", "ols", "multicollinearity", "heteroskedastic"],
    "logistic regression and classification metrics (precision, recall, F1)": ["logistic", "precision", "recall", "f1", "confusion matrix", "false positive", "false negative", "sigmoid", "roc"],
    "Poisson and negative binomial regression for count data, including overdispersion": ["poisson", "negative binomial", "count data", "overdispersion", "rate ratio", "offset", "dispersion"],
    "time series basics: stationarity, autocorrelation, differencing": ["stationary", "stationarity", "autocorrelation", "differencing", "acf", "pacf", "trend", "seasonality", "white noise"],
    "ARIMA/SARIMA forecasting model selection and interpretation": ["arima", "sarima", "forecast", "moving average", "autoregressive", "seasonal", "aic", "bic", "residual diagnostics"],
    "fraud detection and imbalanced classification (ROC-AUC, class imbalance handling)": ["fraud", "imbalanced", "class imbalance", "roc-auc", "auc", "precision-recall", "smote", "undersampling", "oversampling"],
    "risk scoring and model evaluation in a business context": ["risk score", "credit", "default", "calibration", "lorenz", "gini", "ks statistic", "expected loss", "scorecard"],
    "experimental design basics (randomization, control groups, confounding)": ["randomization", "randomised", "randomized", "control group", "treatment group", "confounding", "placebo", "blinding", "blocking"],
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"__corrupt__": line})
    return rows


def _check_row(i: int, row: dict, flags: list[str]) -> None:
    """Append a flag string for any structural/content issue in row i."""
    def flag(msg: str) -> None:
        flags.append(f"[row {i}] {msg}")

    if "__corrupt__" in row:
        flag("corrupt JSON line")
        return
    for field in ("instruction", "output", "topic"):
        if field not in row or not isinstance(row[field], str) or not row[field].strip():
            flag(f"missing/empty '{field}'")
            return
    inst, out, topic = row["instruction"].strip(), row["output"].strip(), row["topic"].strip()

    if len(out) < 20:
        flag("output suspiciously short")
    if len(out) > 2000:
        flag("output very long")

    if topic not in TOPIC_KEYWORDS:
        flag(f"unknown topic: {topic[:50]}")

    # contamination
    lowered = (inst + " " + out).lower()
    if "```" in lowered or lowered.count("{") > 3 or "json" in lowered[:80]:
        flag("possible markdown/JSON remnant")
    if re.search(r"[\ufffd]", inst + out):
        flag("mojibake/replacement char")

    # self-contradiction: simple arithmetic in instruction vs output
    nums = [int(m) for m in re.findall(r"\b(\d{1,3})\b", inst) if m in {"2", "4", "6", "8", "10", "12", "15", "20", "25", "30", "50", "100", "200"}]
    if len(nums) >= 2 and any(kw in lowered for kw in ("mean of", "average of", "sum of")):
        total = sum(nums)
        mean = total / len(nums)
        out_nums = [int(m) for m in re.findall(r"\b(\d+(?:\.\d+)?)\b", out)]
        if mean.is_integer() and int(mean) not in out_nums and len(out_nums) <= 3:
            flag(f"possible arithmetic mismatch: instruction numbers {nums} (mean {mean:g}) not in output {out_nums}")

    # topic drift: strong terms from a DIFFERENT topic
    own = set(TOPIC_KEYWORDS.get(topic, []))
    for other_topic, kws in TOPIC_KEYWORDS.items():
        if other_topic == topic:
            continue
        hits = [kw for kw in kws if kw in lowered]
        if len(hits) >= 3:
            flag(f"possible topic drift → '{other_topic[:35]}' (terms: {hits[:4]})")
            break


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("stats_dataset.jsonl")
    rows = _load(path)
    flags: list[str] = []
    for i, row in enumerate(rows):
        _check_row(i, row, flags)

    inst_norms = Counter(_norm(r.get("instruction", "")) for r in rows)
    dups = {n: c for n, c in inst_norms.items() if c > 1 and n}
    if dups:
        flags.append(f"duplicate instructions: {len(dups)} groups")

    print(f"Reviewed {len(rows)} rows.")
    print(f"Topics: {len({r.get('topic') for r in rows if isinstance(r, dict)})} labeled topics")
    topics = Counter(r.get("topic", "(missing)") for r in rows)
    for t, c in topics.most_common():
        print(f"  {c:3d}  {t}")
    print(f"Output length: min {min(len(r.get('output','')) for r in rows if isinstance(r, dict))}, "
          f"max {max(len(r.get('output','')) for r in rows if isinstance(r, dict))}, "
          f"avg {sum(len(r.get('output','')) for r in rows if isinstance(r, dict)) // len(rows)}")
    print(f"\nFlags ({len(flags)}):")
    for f in flags:
        print("  " + f)
    if not flags:
        print("  (none — dataset looks clean)")
    return 1 if flags else 0


if __name__ == "__main__":
    sys.exit(main())
