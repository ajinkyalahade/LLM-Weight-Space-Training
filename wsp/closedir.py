"""A "done" direction for the end-of-thinking token, and an edit that adds it to that token's output row.

    python -I wsp/closedir.py measure --run runs/<name> --model <dir> --pass-dir <gen pass> [<more>] \
        --data <problems.jsonl> --protocol <name> --out <features .npz>
    python -I wsp/closedir.py make --features <.npz> --out <edit dir> [--beta 0 2 4 ...] [--use <beta>]

measure: the model reads each answer of the given passes (teacher-forced, prompt + answer) and, at every line-end
token inside the thinking, saves the final hidden state h (after the last norm: the input of lm_head) and
log p(</think>) at the next position. A position is "closing" when the next token really is </think>. One ledger
line (type "forward").

make: d = mean h at closing positions - mean h at all other thinking line ends (answers weighted equally), scaled to
unit length. The edit adds beta * d to the </think> row of lm_head, so that token's logit rises by beta * (h . d):
more where the state looks like the states the model closes in, less (or lower) elsewhere. For each beta it simulates,
on the measured answers (teacher-forced, so only an estimate), the chance that thinking would have closed before
the place it really closed (closed answers: "early") and the chance a cut answer would have closed by 8k / 12k tokens
("rescued by"). With --use, it writes the edit folder (tensors/lm_head.weight.npz: rows, values; edit.json).
"""
import argparse
import gzip
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CLOSE_ID = 9          # "</think>" in the MiniCPM5 tokenizer (checked 2026-11-21)
TENSOR = "lm_head.weight"


def measure(a):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    import lm
    proto = lm.PROTOCOLS[a.protocol]
    rows = {}
    for p in a.data:
        for line in open(p):
            r = json.loads(line)
            rows[r["id"]] = r
    model, tok = lm.load(a.model)
    assert tok.convert_tokens_to_ids("</think>") == CLOSE_ID
    ends = np.array([tok.decode([i]).endswith("\n") for i in range(len(tok))])
    rng = np.random.default_rng(0)
    H, LP, POS, ANS, CL, W, LE, RG, meta = [], [], [], [], [], [], [], [], []
    t0 = time.time()
    for pd in a.pass_dir:
        for line in gzip.open(os.path.join(pd, "answers.jsonl.gz")):
            r = json.loads(line)
            prompt = list(lm.prompt_ids(tok, rows[r["id"]]["problem"], proto))
            ans = tok.encode(r["text"], add_special_tokens=False)
            close_at = ans.index(CLOSE_ID) if CLOSE_ID in ans else None
            forced = bool(r.get("forced_close"))
            think = ans[:close_at] if close_at is not None else ans
            body = ans if (a.answer_region and close_at is not None) else think
            # every line end in the thinking, the first --head positions, the position right before </think>, and a
            # random --sample share of the other positions (weight 1/sample in the simulation). m3a (2026-11-21)
            # measured only line ends and fired mid-sentence: every kind of position has to be in the calibration.
            # With --answer-region, the same for the answer after </think> (m3b, 2026-11-21, fired there again and
            # again). A forced close (wsp/close.py) gives only its answer region: its thinking is a copy of an answer
            # already measured, and the position before its close is not one the model chose to close at.
            wgt = {}
            for i in range(len(body)):
                in_answer = close_at is not None and i >= close_at
                if forced and not in_answer:
                    continue
                first = (i - close_at < 64) if in_answer else (i < a.head)
                if ends[body[i]] or first or (close_at is not None and i == close_at - 1):
                    wgt[i] = 1.0
                elif rng.random() < a.sample:
                    wgt[i] = 1.0 / a.sample
            want = sorted(wgt)
            seq = prompt + ans[:len(body) + 1]
            cache = make_prompt_cache(model)
            hs = {}
            for c0 in range(0, len(seq), 2048):
                h = model.model(mx.array(seq[c0:c0 + 2048])[None], cache=cache)[0]
                loc = [i for i in want if c0 <= len(prompt) + i < c0 + 2048]
                if loc:
                    sel = h[mx.array([len(prompt) + i - c0 for i in loc])]
                    lp = model.lm_head(sel)
                    lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
                    mx.eval(sel, lp)
                    for k, i in enumerate(loc):
                        hs[i] = (np.array(sel[k].astype(mx.float32)), float(lp[k, CLOSE_ID]))
                mx.eval([c.state for c in cache])
            j = len(meta)
            for i in sorted(hs):
                H.append(hs[i][0].astype(np.float16))
                LP.append(hs[i][1])
                POS.append(i)
                ANS.append(j)
                CL.append(close_at is not None and i == close_at - 1 and not forced)
                W.append(wgt[i])
                LE.append(bool(ends[body[i]]))
                RG.append(close_at is not None and i >= close_at)
            meta.append({"id": r["id"], "source": r["source"], "sample": r.get("sample"), "outcome": r["outcome"],
                         "tokens": r.get("tokens"), "close_at": close_at, "forced": forced})
            del cache
            mx.clear_cache()
            print(f"{time.time() - t0:6.0f} s  {len(meta)} answers  {len(H)} positions", flush=True)
    np.savez(a.out, H=np.stack(H), logp=np.array(LP, np.float32), pos=np.array(POS), ans=np.array(ANS),
             closing=np.array(CL), weight=np.array(W, np.float32), lineend=np.array(LE), answer=np.array(RG),
             meta=json.dumps(meta))
    summary = {"type": "forward", "label": "closedir_measure", "model": a.model, "passes": a.pass_dir,
               "answers": len(meta), "positions": len(H), "closing_positions": int(sum(CL)), "out": a.out,
               "seconds": round(time.time() - t0, 1)}
    k = lm.next_pass(a.run, "fwd", "closedir")
    with open(os.path.join(a.run, "passes.jsonl"), "a") as f:
        f.write(json.dumps({"pass": k, **summary, "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    print(json.dumps(summary, indent=1))


def late_direction(H, pos, closing, lineend, late_from, shrink, answer):
    """Line ends at or after `late_from` thinking tokens vs every other measured position (m3b, 2026-11-21): the
    cut answers never look "done" (whitened done-direction), but h encodes how long the thinking has run. Positions
    in the answer after </think> are in the "other" class (m3c)."""
    A = lineend & (pos >= late_from) & ~closing & ~answer
    B = ~A & ~closing
    muA, muB = H[A].mean(0), H[B].mean(0)
    X = H[B]
    X = X[np.random.default_rng(0).permutation(len(X))[:20000]] - muB
    S = X.T @ X / len(X)
    d = np.linalg.solve(S + shrink * np.mean(np.diag(S)) * np.eye(len(S)), muA - muB)
    return d / np.linalg.norm(d)


def simulate2(lp, delta, pos, ans, closing, w, lineend, meta, label, answer):
    """As simulate, for a given change of the </think> logit at each measured position. stray_close_in_answer: chance
    of at least one more </think> in the answer section (answers whose answer section was measured)."""
    l = lp + delta
    rest = np.log1p(-np.minimum(np.exp(lp), 1 - 1e-7))
    p = np.exp(l - np.logaddexp(rest, l)) * w
    out = {"edit": label, "stray_close_in_answer": [], "midline_close_any": [], "early_close_in_closed": [],
           "closed_before_6k": []}
    marks = (8192, 10240, 12288, 14336)
    for k in marks:
        out[f"cut_closed_by_{k}"] = []
    for j, m in enumerate(meta):
        if (answer & (ans == j)).any():
            out["stray_close_in_answer"].append(1 - np.exp(-p[answer & (ans == j)].sum()))
        if m.get("forced"):
            continue
        sel = (ans == j) & ~closing & ~answer
        q, ps, le = p[sel], pos[sel], lineend[sel]
        out["midline_close_any"].append(1 - np.exp(-q[~le].sum()))
        out["closed_before_6k"].append(1 - np.exp(-q[ps < 6000].sum()))
        if m["close_at"] is not None:
            out["early_close_in_closed"].append(1 - np.exp(-q[ps < m["close_at"] - 1].sum()))
        else:
            for k in marks:
                out[f"cut_closed_by_{k}"].append(1 - np.exp(-q[ps < k].sum()))
    return {k: ((round(float(np.mean(v)), 3) if v else None) if isinstance(v, list) else v) for k, v in out.items()}


def simulate(lp, s, pos, ans, closing, w, lineend, meta, beta):
    """Teacher-forced estimate under the edit. Expected number of closes E = sum of weight * p(</think>) over the
    measured positions (sampled positions stand for 1/sample positions); chance of at least one close = 1 - exp(-E).
    early: closed answers, before their real close. midline: all answers, positions that are not line ends (a close
    there cuts a sentence). by 8k / 12k: cut answers."""
    l = lp + beta * s                                  # new logit minus old log-normaliser
    rest = np.log1p(-np.minimum(np.exp(lp), 1 - 1e-7))   # log of the other tokens' total probability
    p = np.exp(l - np.logaddexp(rest, l)) * w
    early, mid, resc8, resc12 = [], [], [], []
    for j, m in enumerate(meta):
        sel = (ans == j) & ~closing
        q, ps, le = p[sel], pos[sel], lineend[sel]
        mid.append(1 - np.exp(-q[~le].sum()))
        if m["close_at"] is not None:
            early.append(1 - np.exp(-q[ps < m["close_at"] - 1].sum()))
        else:
            resc8.append(1 - np.exp(-q[ps < 8192].sum()))
            resc12.append(1 - np.exp(-q[ps < 12288].sum()))
    f = lambda v: round(float(np.mean(v)), 3) if v else None  # noqa: E731
    return {"beta": beta, "midline_close_any": f(mid), "early_close_in_closed": f(early), "cut_closed_by_8k": f(resc8),
            "cut_closed_by_12k": f(resc12), "n_closed": len(early), "n_cut": len(resc8)}


def direction(H, ans, closing, n, shrink, skip=None):
    """Mean difference (answers weighted equally), whitened by the other positions' covariance when shrink > 0."""
    keep = ans != skip
    negs = [H[(ans == j) & ~closing] for j in range(n) if j != skip and ((ans == j) & ~closing).any()]
    mu_neg = np.stack([x.mean(0) for x in negs]).mean(0)
    d = H[closing & keep].mean(0) - mu_neg
    if shrink > 0:
        X = H[~closing & keep]
        X = X[np.random.default_rng(0).permutation(len(X))[:20000]] - mu_neg
        S = X.T @ X / len(X)
        d = np.linalg.solve(S + shrink * np.mean(np.diag(S)) * np.eye(len(S)), d)
    return d / np.linalg.norm(d)


def make(a):
    z = np.load(a.features)
    H, lp, pos, ans, closing = z["H"].astype(np.float32), z["logp"], z["pos"], z["ans"], z["closing"]
    w = z["weight"] if "weight" in z else np.ones(len(H), np.float32)
    lineend = z["lineend"] if "lineend" in z else np.ones(len(H), bool)
    answer = z["answer"] if "answer" in z else np.zeros(len(H), bool)
    print(f"answer-section positions: {int(answer.sum())}")
    meta = json.loads(str(z["meta"]))
    n = len(meta)
    d = direction(H, ans, closing, n, a.shrink)
    s = H @ d
    print(f"closing positions {int(closing.sum())}, other {int((~closing).sum())}, answers {n}")
    print(f"h.d  closing: mean {s[closing].mean():.2f} sd {s[closing].std():.2f} | other: mean {s[~closing].mean():.2f}"
          f" sd {s[~closing].std():.2f}")
    print(f"log p(</think>) closing: median {np.median(lp[closing]):.2f} | other: median {np.median(lp[~closing]):.2f}")
    # held out: each closing position scored with d fitted without its own answer
    for j in sorted(set(ans[closing].tolist())):
        dj = direction(H, ans, closing, n, a.shrink, skip=j)
        sj = H[ans == j] @ dj
        cl, le = closing[ans == j], lineend[ans == j]
        print(f"  held out answer {j} ({meta[j]['outcome']}, close at {meta[j]['close_at']}): h.d closing "
              f"{sj[cl].mean():.1f} | its line ends: median {np.median(sj[~cl & le]):.1f}, max "
              f"{sj[~cl & le].max():.1f} | mid-line: median {np.median(sj[~le]) if (~le).any() else np.nan:.1f}, "
              f"max {sj[~le].max() if (~le).any() else np.nan:.1f}")
    if (~lineend).any():
        print("all mid-line positions: h.d percentiles 50/99/99.9/max:",
              np.percentile(s[~lineend], [50, 99, 99.9, 100]).round(1).tolist(), f"(n={int((~lineend).sum())})")
    cut = np.array([meta[j]["close_at"] is None for j in ans])
    print("cut answers, line ends after 8k: h.d percentiles 50/90/99:",
          np.percentile(s[cut & (pos >= 8192)], [50, 90, 99]).round(1).tolist())
    if a.late_from:
        dl = late_direction(H, pos, closing, lineend, a.late_from, a.shrink, answer)
        sl = H @ dl
        print("late direction h.d_late at thinking line ends by position:",
              [(lo, round(float(np.median(sl[lineend & ~closing & ~answer & (pos >= lo) & (pos < lo + 2000)])), 2))
               for lo in range(0, 16000, 2000)], "| mid-line median", round(float(np.median(sl[~lineend & ~answer])), 2),
              "| closing", sl[closing].round(1).tolist())
        if answer.any():
            print("  answer section: h.d_late percentiles 50/99/99.9/max",
                  np.percentile(sl[answer], [50, 99, 99.9, 100]).round(1).tolist(),
                  "| h.d_done", np.percentile(s[answer], [50, 99, 99.9, 100]).round(1).tolist())
        table = [simulate2(lp, bd * s + bl * sl, pos, ans, closing, w, lineend, meta, f"done {bd} + late {bl}", answer)
                 for bd in a.beta for bl in a.beta_late]
    else:
        table = [simulate(lp, s, pos, ans, closing, w, lineend, meta, b) for b in a.beta]
    for t in table:
        print(t)
    if a.use is None:
        return
    if a.late_from:
        d = a.use * d + a.use_late * dl          # the full change to the row (stored as d.npy; beta below is 1)
        a.use = 1.0
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import st
    model_dir = a.model
    row = st.Dir(model_dir).f32(TENSOR)[CLOSE_ID].copy()
    new = row + a.use * d
    os.makedirs(os.path.join(a.out, "tensors"), exist_ok=True)
    np.savez(os.path.join(a.out, "tensors", TENSOR + ".npz"), rows=np.array([CLOSE_ID], np.int32),
             values=new[None].astype(np.float32))
    np.save(os.path.join(a.out, "d.npy"), d.astype(np.float32))
    json.dump({"op": "add_direction_to_output_row", "tensor": TENSOR, "row": CLOSE_ID, "token": "</think>",
               "beta": a.use, "shrink": a.shrink, "late_from": a.late_from, "beta_done": a.beta_done_used,
               "beta_late": a.use_late, "features": a.features, "row_norm_before": float(np.linalg.norm(row)),
               "row_norm_after": float(np.linalg.norm(new)), "changed_values": int((row != new).sum()),
               "simulation": table, "model": model_dir}, open(os.path.join(a.out, "edit.json"), "w"), indent=1)
    print("wrote", a.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--run", required=True)
    m.add_argument("--model", required=True)
    m.add_argument("--pass-dir", nargs="+", required=True)
    m.add_argument("--data", nargs="+", required=True)
    m.add_argument("--protocol", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--head", type=int, default=256)
    m.add_argument("--sample", type=float, default=0.1)
    m.add_argument("--answer-region", action="store_true", help="also measure the answer after </think> (m3c)")
    k = sub.add_parser("make")
    k.add_argument("--shrink", type=float, default=0.0, help="LDA: d = (S + shrink*mean(diag S)*I)^-1 (mu_close - mu_other); 0 = plain mean difference")
    k.add_argument("--features", required=True)
    k.add_argument("--model", default="models/MiniCPM5-2B")
    k.add_argument("--beta", type=float, nargs="+", default=[0, 1, 2, 3, 4, 6, 8, 12, 16])
    k.add_argument("--use", type=float, default=None)
    k.add_argument("--late-from", type=int, default=0, help="also fit the late-line-end direction (m3b)")
    k.add_argument("--beta-late", type=float, nargs="+", default=[0])
    k.add_argument("--use-late", type=float, default=0.0)
    k.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "make":
        a.beta_done_used = a.use
    measure(a) if a.cmd == "measure" else make(a)


if __name__ == "__main__":
    main()
