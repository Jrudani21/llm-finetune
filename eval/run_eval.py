"""Base vs fine-tuned eval on held-out stats questions. Stdlib only.
Usage: python run_eval.py --base qwen2.5:7b --tuned qwen25-stats
Scoring is keyword-based: an answer passes if it contains ANY listed stem
(case-insensitive). Coarse by design; read results.json for the raw answers."""
import argparse, json, math, time, urllib.request

def ask(model, q, backend, timeout=120):
    if backend == "fake":
        return "overdispersion " + q if model == "tuned" else q
    if backend == "lmstudio":  # OpenAI-compatible; reasoning off so answers fit the token cap
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": q}],
                           "temperature": 0, "max_tokens": 400, "reasoning_effort": "none"}).encode()
        req = urllib.request.Request("http://127.0.0.1:1234/v1/chat/completions", body,
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)["choices"][0]["message"].get("content") or ""
    body = json.dumps({"model": model, "prompt": q, "stream": False,
                       "options": {"temperature": 0, "num_predict": 400}}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["response"]

def ask_retry(model, q, backend, tries=4):
    for i in range(tries):
        try:
            return ask(model, q, backend)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(10 * (i + 1))  # LM Studio returns transient 500s while swapping models

def passes(answer, keys):
    if answer.startswith("ERROR"):
        return False  # a failed call is never a pass
    a = answer.lower()
    return any(k.lower() in a for k in keys)

def mcnemar_exact(b, c):
    """Two-sided exact p for discordant counts b (base-only) and c (tuned-only)."""
    n = b + c
    if n == 0:
        return 1.0
    p = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tuned", required=True)
    ap.add_argument("--data", default="held_out.jsonl")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--backend", choices=["ollama", "lmstudio", "fake"], default="ollama")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    def run_model(model, name):  # one model at a time: avoids reload thrash on an 8 GB GPU
        out = []
        for r in rows:
            try:
                out.append(ask_retry(model if a.backend != "fake" else name, r["question"], a.backend))
            except Exception as ex:  # a failed call counts as a miss, recorded verbatim
                out.append(f"ERROR: {ex}")
        return out
    ans_b, ans_t = run_model(a.base, "base"), run_model(a.tuned, "tuned")
    res, b, c = [], 0, 0
    for r, xb, xt in zip(rows, ans_b, ans_t):
        pb, pt = passes(xb, r["must_include"]), passes(xt, r["must_include"])
        if not (xb.startswith("ERROR") or xt.startswith("ERROR")):
            b += pb and not pt
            c += pt and not pb
        res.append({"id": r["id"], "topic": r["topic"], "base_pass": pb, "tuned_pass": pt,
                    "base_answer": xb, "tuned_answer": xt})
    n = len(res)
    errs = sum(x["base_answer"].startswith("ERROR") or x["tuned_answer"].startswith("ERROR") for x in res)
    if errs:
        print(f"WARNING: {errs}/{n} rows had a failed call; they score as misses and are excluded from McNemar")
    out = {"n": n, "base_acc": sum(x["base_pass"] for x in res) / n,
           "tuned_acc": sum(x["tuned_pass"] for x in res) / n,
           "base_only": b, "tuned_only": c, "mcnemar_p": mcnemar_exact(b, c), "rows": res}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(f"| model | accuracy |\n|---|---|\n| {a.base} | {out['base_acc']:.0%} |\n"
          f"| {a.tuned} | {out['tuned_acc']:.0%} |\nMcNemar exact p = {out['mcnemar_p']:.4f} "
          f"(base-only {b}, tuned-only {c}, n={n})")

if __name__ == "__main__":
    main()
