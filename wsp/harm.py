"""Do-no-harm check: does an edit hurt what the model could already do outside the edited area?

    python -I wsp/harm.py --run runs/<name> --model <dir> [--edit <edit dir>] --label <name> \
        --mmlu data/<name>/harm/mmlu200.jsonl --texts data/<name>/harm/prose.txt data/<name>/harm/code.txt

Two measurements, both read-only and deterministic (no sampling):
    knowledge   multiple-choice questions asked with thinking OFF; the model's answer is the letter A-D it gives the
                highest probability as the first answer token. Score = share correct.
    text        mean log-probability per token the model gives to each text file (read in pieces of --piece tokens).
                Closer to 0 is better; an edit that damages general language would push it down.
Saved in runs/<name>/passes/fwd_<k>_harm_<label>/summary.json, with every question's prediction; one ledger line.
Compare two such passes (base and edited) with `wsp/report.py harm`.
"""
import argparse
import json
import os
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm  # noqa: E402


def letter_ids(tok):
    ids = []
    for L in "ABCD":
        t = tok.encode(L, add_special_tokens=False)
        assert len(t) == 1, f"letter {L} is not one token: {t}"
        ids.append(t[0])
    return ids


def question_prompt(tok, q):
    body = q["question"].strip() + "\n\n" + "\n".join(f"{L}. {c}" for L, c in zip("ABCD", q["choices"]))
    body += "\n\nAnswer with the letter of the correct option only."
    return tok.apply_chat_template([{"role": "user", "content": body}], add_generation_prompt=True, tokenize=True,
                                   enable_thinking=False)


def text_logp(model, ids, piece):
    total, n = 0.0, 0
    for s in range(0, len(ids) - 1, piece):
        chunk = ids[s:s + piece + 1]
        if len(chunk) < 2:
            break
        lg = model(mx.array([chunk[:-1]]))[0].astype(mx.float32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        got = mx.take_along_axis(lp, mx.array(chunk[1:])[:, None], axis=-1)[:, 0]
        total += float(got.sum())
        n += len(chunk) - 1
        mx.clear_cache()
    return total / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--edit", default=None)
    ap.add_argument("--label", required=True)
    ap.add_argument("--mmlu", required=True)
    ap.add_argument("--texts", nargs="*", default=[])
    ap.add_argument("--piece", type=int, default=1024)
    a = ap.parse_args()

    model, tok = lm.load(a.model, a.edit)
    letters = letter_ids(tok)
    t0 = time.time()
    qs = [json.loads(line) for line in open(a.mmlu)]
    preds = []
    for q in qs:
        lg = model(mx.array([question_prompt(tok, q)]))[0, -1].astype(mx.float32)
        p = np.array(mx.softmax(lg[mx.array(letters)]))
        preds.append({"id": q["id"], "subject": q["subject"], "gold": q["answer"], "pred": "ABCD"[int(p.argmax())],
                      "p_gold": round(float(p["ABCD".index(q["answer"])]), 4)})
    acc = float(np.mean([r["pred"] == r["gold"] for r in preds]))
    texts = {}
    for path in a.texts:
        mean, n = text_logp(model, tok.encode(open(path).read(), add_special_tokens=False), a.piece)
        texts[os.path.basename(path)] = {"tokens": n, "mean_logp": round(mean, 5)}

    k = lm.next_pass(a.run, "fwd", f"harm_{a.label}")
    out_dir = os.path.join(a.run, "passes", f"fwd_{k}_harm_{a.label}")
    os.makedirs(out_dir, exist_ok=True)
    summary = {"pass": k, "type": "harm", "label": f"harm_{a.label}", "model": a.model, "edit": a.edit,
               "mmlu": a.mmlu, "questions": len(qs), "accuracy": round(acc, 4), "texts": texts,
               "seconds": round(time.time() - t0, 1), "peak_gb": round(lm.peak_gb(), 2), "dir": out_dir}
    json.dump({**summary, "predictions": preds}, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    with open(os.path.join(a.run, "passes.jsonl"), "a") as f:
        f.write(json.dumps({**summary, "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
