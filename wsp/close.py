"""Forced close: what an answer cut at the token cap would have scored had its thinking stopped earlier.

    python -I wsp/close.py --run runs/<name> --model <dir> [--edit <dir>] --pass-dir <gen pass> --data <problems.jsonl> \
        --protocol <name> [--at 8192 16384] [--answer-tokens 4096] [--batch 8] [--seed 0] [--label close]

For every answer in --pass-dir that was cut at the cap while still thinking, and every K in --at: keep its first K
answer tokens, cut back to the last paragraph break, append the template's close marker ("\n</think>\n\n"), and let the
model write the answer (at most --answer-tokens, same sampling). The result is graded like a normal answer (for code:
run in the sandbox). Saved in runs/<name>/passes/gen_<k>_<label>/: answers.jsonl.gz and summary.json; one line in
passes.jsonl. This measures the ceiling of an edit that makes the model stop thinking and answer.
"""
import argparse
import gzip
import json
import os
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm  # noqa: E402
import sandbox  # noqa: E402
from mlx_lm.generate import BatchGenerator  # noqa: E402

CLOSE = "\n</think>\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--edit", default=None)
    ap.add_argument("--pass-dir", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--at", type=int, nargs="+", default=[8192, 16384])
    ap.add_argument("--answer-tokens", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="close")
    a = ap.parse_args()

    proto = lm.PROTOCOLS[a.protocol]
    rows = {}
    for p in a.data:
        for line in open(p):
            r = json.loads(line)
            rows[r["id"]] = r
    src = [json.loads(line) for line in gzip.open(os.path.join(a.pass_dir, "answers.jsonl.gz"))]
    cut = [r for r in src if r["outcome"] == "cut_at_max_tokens" and not r["closed_thinking"]]
    k = lm.next_pass(a.run, "gen", a.label)
    out = os.path.join(a.run, "passes", f"gen_{k}_{a.label}")
    os.makedirs(out, exist_ok=True)
    model, tok = lm.load(a.model, a.edit)
    mx.random.seed(a.seed)

    jobs = []
    for r in cut:
        ids = tok.encode(r["text"], add_special_tokens=False)
        for K in a.at:
            text = tok.decode(ids[:K])
            cutpos = text.rfind("\n\n")
            text = (text[:cutpos] if cutpos > len(text) // 2 else text).rstrip() + CLOSE
            prompt = lm.prompt_ids(tok, rows[r["id"]]["problem"], proto)
            jobs.append((r, K, text, list(prompt) + tok.encode(text, add_special_tokens=False)))

    t0 = time.time()
    recs = []
    with open(os.path.join(out, "progress.log"), "a") as log:
        for w0 in range(0, len(jobs), a.batch):
            wave = jobs[w0:w0 + a.batch]
            gen = BatchGenerator(model, stop_tokens=[[t] for t in tok.eos_token_ids], sampler=lm.sampler(proto),
                                 completion_batch_size=a.batch, prefill_batch_size=min(a.batch, 4))
            uids = gen.insert([j[3] for j in wave], [a.answer_tokens] * len(wave))
            where, toks = dict(zip(uids, range(len(wave)))), {i: [] for i in range(len(wave))}
            while rs := gen.next_generated():
                for x in rs:
                    i = where[x.uid]
                    if x.finish_reason != "stop":
                        toks[i].append(x.token)
                    if x.finish_reason is not None:
                        r, K, text, _ = wave[i]
                        tail = tok.decode(toks[i])
                        g = sandbox.grade_code(text + tail, rows[r["id"]]["tests"], x.finish_reason)
                        recs.append({"id": r["id"], "source": r["source"], "at": K, "answer_tokens": len(toks[i]),
                                     "finish": x.finish_reason, **g, "tail": tail, "text": text + tail,
                                     "forced_close": True})
            gen.close()
            del gen
            mx.clear_cache()
            log.write(f"{time.time() - t0:7.0f} s  done {len(recs)}/{len(jobs)}\n")
            log.flush()

    with gzip.open(os.path.join(out, "answers.jsonl.gz"), "wt") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = {}
    for K in a.at:
        rs = [r for r in recs if r["at"] == K]
        by[str(K)] = {"correct": sum(r["correct"] for r in rs), "of": len(rs),
                      "by_source": {s: sum(r["correct"] for r in rs if r["source"] == s)
                                    for s in sorted({r["source"] for r in rs})},
                      "outcomes": {o: sum(r["outcome"] == o for r in rs)
                                   for o in sorted({r["outcome"] for r in rs})}}
    summary = {"pass": k, "type": "close", "label": a.label, "model": a.model, "edit": a.edit,
               "protocol": a.protocol, "source_pass": a.pass_dir, "cut_answers": len(cut), "at": by,
               "answer_tokens": a.answer_tokens, "seconds": round(time.time() - t0, 1), "peak_gb": round(lm.peak_gb(), 2),
               "dir": out}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    with open(os.path.join(a.run, "passes.jsonl"), "a") as f:
        f.write(json.dumps({**summary, "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
