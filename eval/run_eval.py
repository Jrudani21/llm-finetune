"""Base vs fine-tuned eval on held-out stats questions. Stdlib only.
Usage: python run_eval.py --base qwen2.5:7b --tuned qwen25-stats
Scoring is keyword-based: an answer passes if it contains ANY listed stem
(case-insensitive). Coarse by design; read results.json for the raw answers."""
import argparse, json, math, urllib.request

def ask(model, q, backend, timeout=120):
    if backend == "fake":
        return "overdispersion " + q if model == "tuned" else q
    body = json.dumps({"model": model, "prompt": q, "stream": False,
                       "options": {"temperature": 0, "num_predict": 400}}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["response"]

def passes(answer, keys):
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
    ap.add_argument("--backend", choices=["ollama", "fake"], default="ollama")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    res, b, c = [], 0, 0
    for r in rows:
        ans = {m: ask(m if a.backend == "ollama" else n, r["question"], a.backend)
               for m, n in ((a.base, "base"), (a.tuned, "tuned"))}
        pb, pt = passes(ans[a.base], r["must_include"]), passes(ans[a.tuned], r["must_include"])
        b += pb and not pt
        c += pt and not pb
        res.append({"id": r["id"], "topic": r["topic"], "base_pass": pb, "tuned_pass": pt,
                    "base_answer": ans[a.base], "tuned_answer": ans[a.tuned]})
    n = len(res)
    out = {"n": n, "base_acc": sum(x["base_pass"] for x in res) / n,
           "tuned_acc": sum(x["tuned_pass"] for x in res) / n,
           "base_only": b, "tuned_only": c, "mcnemar_p": mcnemar_exact(b, c), "rows": res}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(f"| model | accuracy |\n|---|---|\n| {a.base} | {out['base_acc']:.0%} |\n"
          f"| {a.tuned} | {out['tuned_acc']:.0%} |\nMcNemar exact p = {out['mcnemar_p']:.4f} "
          f"(base-only {b}, tuned-only {c}, n={n})")

if __name__ == "__main__":
    main()
