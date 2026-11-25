"""A generation pass: a model answers problems under a fixed protocol, and the answers are graded.

    python -I wsp/generate.py --run runs/<name> --model <model dir> [--edit <edit dir>] --label original \
        --problems <ids.json | first:N> --data <practice .jsonl> [<more .jsonl>] \
        [--n 1] [--max-tokens 32768] [--batch 32] [--protocol think35] [--seed 0] [--no-ledger]

Writes runs/<name>/passes/gen_<k>_<label>/: answers.jsonl.gz (one line per answer: id, sample, tokens, finish, text
and its grade), summary.json, progress.log (tokens per second as the pass goes). Unless --no-ledger, it appends one
line to runs/<name>/passes.jsonl: that file is the count of forward passes used on this model.

While a pass runs the answers are in answers.jsonl, one line added as each answer finishes. If the pass is interrupted,
the same command continues it. Answers are generated in waves of --batch (see the comment in main). Memory grows with answer length (about 16 KB per token per running answer for
Qwen3.5-0.8B), so the batch size sets the peak: 16 answers at 32,768 tokens is about 11 GB.
"""
import argparse
import gzip
import json
import os
import shutil
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grade  # noqa: E402
import lm  # noqa: E402
import sandbox  # noqa: E402
from mlx_lm.generate import BatchGenerator  # noqa: E402


def load_rows(paths):
    rows = {}
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def pick(spec, rows):
    if spec.startswith("first:"):
        return list(rows)[:int(spec[6:])]
    d = json.load(open(spec))
    return d["problems"] if isinstance(d, dict) else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--edit", default=None, help="a folder written by wsp/edit.py, applied on top of --model")
    ap.add_argument("--label", default="original")
    ap.add_argument("--problems", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--protocol", default="think35")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ledger", action="store_true")
    a = ap.parse_args()

    proto = lm.PROTOCOLS[a.protocol]
    cap = a.max_tokens or proto["max_tokens"]
    rows = load_rows(a.data)
    ids = pick(a.problems, rows)
    k = lm.next_pass(a.run, "gen", a.label)
    out = os.path.join(a.run, "passes", f"gen_{k}_{a.label}" + ("_trial" if a.no_ledger else ""))
    os.makedirs(out, exist_ok=True)

    model, tok = lm.load(a.model, a.edit)
    jobs = [(i, s) for i in ids for s in range(a.n)]

    # Every answer is written the moment it finishes, so a pass that is interrupted loses only the answers still
    # being written; running the same command again continues it.
    plain = os.path.join(out, "answers.jsonl")
    assert not os.path.exists(plain + ".gz"), f"{out} already holds a finished pass"
    setup = {"model": a.model, "edit": a.edit, "protocol": a.protocol, "max_tokens": cap, "jobs": jobs}
    setup_file = os.path.join(out, "setup.json")
    if os.path.exists(setup_file):
        assert json.load(open(setup_file)) == json.loads(json.dumps(setup)), f"{out} was started with other settings"
    json.dump(setup, open(setup_file, "w"))
    have = {}
    if os.path.exists(plain):
        for line in open(plain):
            r = json.loads(line)
            have[(r["id"], r["sample"])] = r
    todo = [j for j, job in enumerate(jobs) if job not in have]
    mx.random.seed(a.seed + len(have))

    mx.reset_peak_memory()
    t0 = last = time.time()
    total = since = 0
    done = len(have)
    log = open(os.path.join(out, "progress.log"), "a")
    if have:
        log.write(f"continuing: {len(have)} answers already saved\n")
    f = open(plain, "a")
    # Waves: a fresh generator for each group of --batch answers, which runs until all of them have finished. mlx-lm
    # can also top up a running batch, but a newcomer is then padded to the length of the longest answer already
    # running (up to 32k empty positions), which is slow, and on 2026-11-14 a top-up of 14 answers into a batch at
    # 28k tokens was followed within 80 s by "[metal::malloc] Resource limit (499000) exceeded".
    for w0 in range(0, len(todo), a.batch):
        wave = todo[w0:w0 + a.batch]
        prompts = [lm.prompt_ids(tok, rows[jobs[j][0]]["problem"], proto) for j in wave]
        gen = BatchGenerator(model, stop_tokens=[[t] for t in tok.eos_token_ids], sampler=lm.sampler(proto),
                             completion_batch_size=a.batch, prefill_batch_size=min(a.batch, 8))
        uids = gen.insert(prompts, [cap] * len(prompts),
                          logits_processors=[lm.processors(proto, len(p)) for p in prompts])
        where = dict(zip(uids, wave))
        toks = {j: [] for j in wave}
        log.write(f"{time.time() - t0:7.0f} s  wave {w0 // a.batch + 1} of {-(-len(todo) // a.batch)}: "
                  f"{len(wave)} answers\n")
        while rs := gen.next_generated():
            for r in rs:
                j = where[r.uid]
                if r.finish_reason != "stop":
                    toks[j].append(r.token)
                    total += 1
                    since += 1
                if r.finish_reason is not None:
                    i, s = jobs[j]
                    t = toks.pop(j)
                    text = tok.decode(t)
                    graded = (sandbox.grade_code(text, rows[i]["tests"], r.finish_reason) if "tests" in rows[i]
                              else grade.grade(text, rows[i]["answer"], r.finish_reason))   # code: run its tests
                    rec = {"id": i, "source": rows[i]["source"], "sample": s, "tokens": len(t),
                           "finish": r.finish_reason, "gold": rows[i]["answer"], **graded, "text": text}
                    have[(i, s)] = rec
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    done += 1
            now = time.time()
            if now - last >= 30:
                running = [len(t) for t in toks.values() if t]
                log.write(f"{now - t0:7.0f} s  done {done}/{len(jobs)}  tokens {total}  "
                          f"{since / (now - last):7.1f} tokens/s  running {len(running)} "
                          f"(longest {max(running, default=0)})  peak {lm.peak_gb():.1f} GB\n")
                log.flush()
                last, since = now, 0
        gen.close()
        del gen, where, toks
        mx.clear_cache()
    f.close()
    seconds = time.time() - t0

    recs = [have[job] for job in jobs]
    with open(plain, "rb") as src, gzip.open(plain + ".gz", "wb") as dst:      # looping text compresses very well
        shutil.copyfileobj(src, dst)
    os.remove(plain)

    n = len(recs)
    acc = lambda rs_: round(sum(r["correct"] for r in rs_) / max(1, len(rs_)), 4)  # noqa: E731
    count = lambda o: sum(r["outcome"] == o for r in recs)  # noqa: E731
    summary = {
        "pass": k, "type": "generate", "label": a.label, "model": a.model, "protocol": a.protocol, "max_tokens": cap,
        "edit": a.edit, "seed": a.seed, "batch": a.batch, "generations": n, "accuracy": acc(recs),
        "accuracy_by_source": {s: acc([r for r in recs if r["source"] == s]) for s in sorted({r["source"] for r in recs})},
        "mean_tokens": round(sum(r["tokens"] for r in recs) / n),
        "cut_share": round(count("cut_at_max_tokens") / n, 4),
        "failures": {o: count(o) for o in ("wrong_answer", "no_answer_finished", "cut_at_max_tokens")},
        "loop_end_among_cut": sum(r["loop_end"] for r in recs if r["outcome"] == "cut_at_max_tokens"),
        "seconds": round(seconds, 1), "tokens_per_second": round(total / max(seconds, 1e-9), 1),
        "peak_gb": round(lm.peak_gb(), 2), "answers_from_an_earlier_session": n - len(todo), "dir": out,
    }
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    if not a.no_ledger:
        with open(os.path.join(a.run, "passes.jsonl"), "a") as f:
            f.write(json.dumps({**summary, "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
