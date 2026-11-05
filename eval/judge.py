"""LLM-judge pass over results.json from run_eval.py. Stdlib only.
Usage: python judge.py results_base_vs_base.json --judge openai/gpt-oss-20b
The judge must be a different model than the two being compared. Scores 1-5 against
the question's key points; paired sign test on per-question score differences."""
import argparse, json, re, urllib.request
from run_eval import mcnemar_exact

PROMPT = """You grade answers to applied-statistics questions.
Question: {q}
Reference answer: {k}
Answer to grade:
\"\"\"{a}\"\"\"
Score 1-5: 5 = correct and covers the key points; 3 = partly correct or vague; 1 = wrong or off-topic.
Reply with only the digit."""

def grade(judge, q, keys, answer):
    body = json.dumps({"model": judge, "temperature": 0, "max_tokens": 800,
                       "reasoning_effort": "low",
                       "messages": [{"role": "user", "content": PROMPT.format(q=q, k=", ".join(keys), a=answer[:1500])}]}).encode()
    req = urllib.request.Request("http://127.0.0.1:1234/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        text = json.load(r)["choices"][0]["message"].get("content") or ""
    m = re.search(r"[1-5]", text)
    return int(m.group()) if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--data", default="held_out.jsonl")
    ap.add_argument("--judge", required=True)
    ap.add_argument("--out", default="judged.json")
    a = ap.parse_args()
    d = json.load(open(a.results, encoding="utf-8"))
    qs = {json.loads(l)["id"]: json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()}
    for side in ("base", "tuned"):  # one side at a time, answers only, judge never sees model names
        for r in d["rows"]:
            q = qs[r["id"]]
            try:
                r[side + "_score"] = grade(a.judge, q["question"], [q.get("reference") or ", ".join(q["must_include"])], r[side + "_answer"])
            except Exception as ex:
                r[side + "_score"] = None
                r[side + "_judge_error"] = str(ex)
    ok = [r for r in d["rows"] if r["base_score"] and r["tuned_score"]]
    n = len(ok)
    bw = sum(r["base_score"] > r["tuned_score"] for r in ok)
    tw = sum(r["tuned_score"] > r["base_score"] for r in ok)
    d["judge"] = {"model": a.judge, "n_scored": n, "of": len(d["rows"]),
                  "base_mean": sum(r["base_score"] for r in ok) / n,
                  "tuned_mean": sum(r["tuned_score"] for r in ok) / n,
                  "base_wins": bw, "tuned_wins": tw, "ties": n - bw - tw,
                  "sign_test_p": mcnemar_exact(bw, tw)}
    json.dump(d, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(d["judge"], indent=1))

if __name__ == "__main__":
    main()
