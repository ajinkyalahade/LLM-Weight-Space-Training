"""Matched pairs: continue the same text several times from just before a loop starts, and label each continuation.

    python -I wsp/branch.py --run runs/<name> --model <dir> --source <probe request .json[.gz]> \
        [--offsets -300 0] [--n 8] [--length 300] [--batch 16] [--seed 0] [--label branch] \
        [--rep 0.85] [--prod 0.55] [--window 150] --out <probe request to write>

Run it only when no other generation is using the GPU: on 2026-11-15 it ran out of GPU memory next to a full-length
pass (32 prefixes of 4-10k tokens, 8 read at once).

--source is a probe request written by wsp/loops.py (each case: tokens = prompt + answer, onset, groups). For every
case and every offset, the prefix is tokens[:onset + offset]; the model continues it --n times for --length tokens
under the case's protocol (temperature, top-p, top-k, presence penalty; the penalty counts the answer tokens
already in the prefix, as it would have in the original answer).

A continuation is "rep" (it loops) when at least --rep of its tokens repeat an 8-gram seen earlier in the prefix or
in itself, and "prod" (it does not) when at most --prod do. Saved in runs/<name>/passes/gen_<k>_<label>/:
branches.jsonl.gz (every continuation with its repeat share) and summary.json; one line in passes.jsonl.

--out: a probe request with one case per labelled continuation, named "<case>@<offset>__<rep|prod>_<i>", whose only
group "cont" is its first --window tokens. Only prefixes that produced at least one of each label are kept, so every
looping continuation has a non-looping partner from the identical prefix. wsp/localise.py --matched reads it.
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
from mlx_lm.generate import BatchGenerator  # noqa: E402


def repeat_share(prefix, cont, n=8):
    seen = {tuple(prefix[i:i + n]) for i in range(len(prefix) - n + 1)}
    seq, hits = list(prefix[-(n - 1):]) + list(cont), 0
    for t in range(n - 1, len(seq)):
        g = tuple(seq[t - n + 1:t + 1])
        hits += g in seen
        seen.add(g)
    return hits / max(1, len(cont))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--edit", default=None)
    ap.add_argument("--source", required=True)
    ap.add_argument("--offsets", type=int, nargs="+", default=[-300, 0])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--length", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="branch")
    ap.add_argument("--protocol", default="think35")
    ap.add_argument("--rep", type=float, default=0.85)
    ap.add_argument("--prod", type=float, default=0.55)
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    src = json.load(gzip.open(a.source, "rt") if a.source.endswith(".gz") else open(a.source))
    proto = lm.PROTOCOLS[a.protocol]
    model, tok = lm.load(a.model, a.edit)
    jobs = []                                  # (case name, offset, sample, prefix, prompt length)
    for c in src["cases"]:
        P = c["groups"]["productive"][0][0] - 20          # wsp/loops.py starts the productive group at P + 20
        for off in a.offsets:
            prefix = c["tokens"][:c["onset"] + off]
            jobs += [(c["name"], off, s, prefix, P) for s in range(a.n)]

    mx.random.seed(a.seed)
    t0, conts = time.time(), [None] * len(jobs)
    for w0 in range(0, len(jobs), a.batch):
        wave = list(range(w0, min(len(jobs), w0 + a.batch)))
        gen = BatchGenerator(model, stop_tokens=[[t] for t in tok.eos_token_ids], sampler=lm.sampler(proto),
                             completion_batch_size=a.batch, prefill_batch_size=2)     # prefixes are long (4-10k)
        uids = gen.insert([jobs[j][3] for j in wave], [a.length] * len(wave),
                          logits_processors=[lm.processors(proto, jobs[j][4]) for j in wave])
        where, toks = dict(zip(uids, wave)), {j: [] for j in wave}
        while rs := gen.next_generated():
            for r in rs:
                if r.finish_reason != "stop":
                    toks[where[r.uid]].append(r.token)
        for j in wave:
            conts[j] = toks[j]
        gen.close()
        del gen
        mx.clear_cache()
        print(f"{min(len(jobs), w0 + a.batch)}/{len(jobs)} continuations, {time.time() - t0:.0f} s", flush=True)

    k = lm.next_pass(a.run, "gen", a.label)                 # the folder is made only once the work has succeeded
    out_dir = os.path.join(a.run, "passes", f"gen_{k}_{a.label}")
    os.makedirs(out_dir, exist_ok=True)
    recs = []
    for (name, off, s, prefix, P), cont in zip(jobs, conts):
        share = repeat_share(prefix, cont)
        label = "rep" if share >= a.rep else "prod" if share <= a.prod else "mixed"
        recs.append({"case": name, "offset": off, "sample": s, "repeat_share": round(share, 3), "label": label,
                     "tokens": len(cont), "text": tok.decode(cont)})
    with gzip.open(os.path.join(out_dir, "branches.jsonl.gz"), "wt") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cases, table = [], {}
    for (name, off, s, prefix, P), cont, r in zip(jobs, conts, recs):
        table.setdefault((name, off), []).append((r["label"], s, prefix, cont))
    kept = 0
    for (name, off), items in table.items():
        labels = [x[0] for x in items]
        print(f"{name:24} offset {off:5d}: {labels.count('rep')} loop, {labels.count('prod')} do not, "
              f"{labels.count('mixed')} unclear")
        if "rep" not in labels or "prod" not in labels:
            continue
        kept += 1
        for label, s, prefix, cont in items:
            if label != "mixed" and len(cont) >= a.window:
                cases.append({"name": f"{name}@{off}__{label}_{s}", "tokens": list(prefix) + list(cont),
                              "groups": {"cont": [[len(prefix), len(prefix) + a.window]]}})
    json.dump({"layers": src.get("layers", []), "source": a.source, "matched": True, "cases": cases},
              open(a.out, "w"))
    n_tok = sum(len(c) for c in conts)
    summary = {"pass": k, "type": "generate", "label": a.label, "model": a.model, "edit": a.edit,
               "protocol": a.protocol, "source": a.source, "offsets": a.offsets, "n": a.n, "length": a.length,
               "continuations": len(recs), "rep": sum(r["label"] == "rep" for r in recs),
               "prod": sum(r["label"] == "prod" for r in recs), "prefixes_with_both": kept,
               "tokens": n_tok, "seconds": round(time.time() - t0, 1), "peak_gb": round(lm.peak_gb(), 2),
               "dir": out_dir}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    with open(os.path.join(a.run, "passes.jsonl"), "a") as f:
        f.write(json.dumps({**summary, "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"wrote {a.out}: {len(cases)} cases from {kept} prefixes")


if __name__ == "__main__":
    main()
