"""Find where each answer of a generation pass starts repeating itself, and build a probe request from it.

    python -I wsp/loops.py --pass-dir runs/<name>/passes/gen_<k>_<label> --model <dir> --data <practice .jsonl> ... \
        [--out <request.json>] [--cases 8] [--skip 0] [--max-tokens 12000] [--layers 15 19]

A token is "repeated" when the 8 tokens ending at it already occurred earlier in the same answer. The loop onset is
the first token from which every following window of 200 tokens is at least 90% repeated. An answer has a loop when
such an onset exists and at least 400 tokens follow it.

With --out, writes a request for wsp/probe.py: the answers with the earliest onsets, each cut 1,100 tokens after
its onset, with two groups of positions:
    productive   from 20 tokens into the answer to 150 tokens before the onset
    repetitive   from 100 tokens after the onset, for up to 1,000 tokens
Both groups come from the same answer, so the comparison is the model with itself on the same problem.
"""
import argparse
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def repeated(tokens, n=8):
    seen, flag = set(), np.zeros(len(tokens), bool)
    for t in range(n - 1, len(tokens)):
        g = tuple(tokens[t - n + 1:t + 1])
        if g in seen:
            flag[t] = True
        else:
            seen.add(g)
    return flag


def onset(flag, window=200, share=0.9, tail=400):
    """Position where the loop starts, or None."""
    if len(flag) < window + tail:
        return None
    c = np.concatenate([[0], np.cumsum(flag)])
    roll = (c[window:] - c[:-window]) / window              # roll[i] = share repeated in flag[i:i + window]
    bad = np.where(roll < share)[0]
    start = int(bad.max()) + 1 if bad.size else 0
    return start if len(flag) - start >= tail else None


def read(pass_dir):
    path = os.path.join(pass_dir, "answers.jsonl")
    f = gzip.open(path + ".gz", "rt") if os.path.exists(path + ".gz") else open(path)
    return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cases", type=int, default=8)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--layers", type=int, nargs="*", default=[])
    ap.add_argument("--protocol", default="think35")
    a = ap.parse_args()
    import lm
    from mlx_lm.utils import load_tokenizer
    from pathlib import Path
    tok = load_tokenizer(Path(a.model))
    proto = lm.PROTOCOLS[a.protocol]
    problems = {}
    for p in a.data:
        for line in open(p):
            r = json.loads(line)
            problems[r["id"]] = r["problem"]

    found = []
    print(f"{'id':28} {'outcome':20} {'tokens':>7} {'onset':>7} {'repeated after onset':>21}")
    for r in read(a.pass_dir):
        ans = tok.encode(r["text"], add_special_tokens=False)
        flag = repeated(ans)
        o = onset(flag)
        after = f"{100 * flag[o:].mean():.0f}%" if o is not None else ""
        print(f"{r['id']:28} {r['outcome']:20} {len(ans):7d} {'' if o is None else o:>7} {after:>21}")
        if o is not None:
            found.append((o, r, ans))
    n = len(read(a.pass_dir))
    onsets = sorted(o for o, _, _ in found)
    print(f"\n{len(found)} of {n} answers have a loop" +
          (f"; onset median {int(np.median(onsets))}, earliest {onsets[0]}, latest {onsets[-1]} tokens" if found else ""))
    if not a.out:
        return
    cases, skipped = [], 0
    for o, r, ans in sorted(found, key=lambda x: x[0]):
        prompt = lm.prompt_ids(tok, problems[r["id"]], proto)
        P, end = len(prompt), min(len(ans), o + 1100)
        if o < 400 or P + end > a.max_tokens or len(cases) >= a.cases:
            continue
        if skipped < a.skip:                    # leave the first --skip eligible answers to another request
            skipped += 1
            continue
        cases.append({"name": f"{r['id']}_s{r['sample']}", "tokens": prompt + ans[:end], "onset": P + o,
                      "groups": {"productive": [[P + 20, P + o - 150]], "repetitive": [[P + o + 100, P + end]]}})
    json.dump({"layers": a.layers, "source": a.pass_dir, "cases": cases}, open(a.out, "w"))
    print(f"wrote {a.out}: {len(cases)} cases, {sum(len(c['tokens']) for c in cases):,} tokens")


if __name__ == "__main__":
    main()
