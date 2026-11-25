"""Graphs and tables for a model's results folder, made only from files that are already saved.

    python -I wsp/report.py weights --run runs/<name> --base <model dir> --other <edited model dir | edit dir> --name <id>
        graphs/<id>_weights.svg: every changed neuron, its push along the edit's direction before and after.

    python -I wsp/report.py passes --run runs/<name> --passes 1,2 --name <id> [--ref "Post, original=gsm8k:6.25,math:14.06,all:11.46"]
        graphs/<id>_accuracy.svg, _outcomes.svg, _lengths.svg, _speed.svg for those passes of runs/<name>/passes.jsonl,
        and a markdown table on standard output (paired difference of the last pass against the first).

    python -I wsp/report.py layers --run runs/<name> --edit runs/<name>/edits/<id> --name <id>
        graphs/<id>_layers.svg: per layer, how large and how consistent the difference between the two probed groups is.

    python -I wsp/report.py probes --run runs/<name> --passes 3,4,5 --names "Authors' edit,My edit" --name <id>
        graphs/<id>_confidence.svg: how much each edited model lowers the log-probability of each group's text.

Graphs go to runs/<name>/graphs/. Nothing here runs a model.
"""
import argparse
import glob
import gzip
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chart  # noqa: E402
import st  # noqa: E402

OUTCOMES = [("correct", "Correct"), ("wrong_answer", "Wrong answer"), ("no_answer_finished", "Finished, no answer"),
            ("cut_at_max_tokens", "Cut off at the token cap")]
SOURCE = {"gsm8k": "GSM8K", "math": "MATH", "aime": "AIME", "lcb_easy": "Easy", "lcb_medium": "Medium", "lcb_hard": "Hard"}
ORDER = list(SOURCE)


def save(run, name, svg):
    os.makedirs(os.path.join(run, "graphs"), exist_ok=True)
    path = os.path.join(run, "graphs", name)
    open(path, "w").write(svg)
    print("wrote", path)


def changed(base, other):
    """{tensor: (columns, new values of those columns)} for an edited model folder or an edit folder."""
    out = {}
    packs = glob.glob(os.path.join(other, "tensors", "*.npz"))
    if packs:
        for p in sorted(packs):
            z = np.load(p)
            out[os.path.basename(p)[:-4]] = (z["columns"], z["values"])
        return out
    o = st.Dir(other)
    for name in base.names:
        if name in o.of and not np.array_equal(base.bits(name), o.bits(name)):
            cols = np.where((base.bits(name) != o.bits(name)).any(0))[0]
            out[name] = (cols, o.f32(name)[:, cols])
    return out


def weights(a):
    base = st.Dir(a.base)
    series, rows = [], []
    for name, (cols, new) in changed(base, a.other).items():
        old = base.f32(name)[:, cols].astype(np.float64)
        d = new.astype(np.float64) - old
        u, s, _ = np.linalg.svd(d, full_matrices=False)
        u = u[:, 0]
        before, after = u @ old, u @ new.astype(np.float64)
        if before.mean() < 0:
            before, after = -before, -after
        layer = re.search(r"layers\.(\d+)\.", name)
        label = f"Layer {layer.group(1)}" if layer else name
        series.append((label, [(float(b), float(x), f"{label}, neuron {c}: {b:+.4f} → {x:+.4f}")
                               for c, b, x in zip(cols, before, after)]))
        rows.append((label, name, len(cols), s[0] ** 2 / (s ** 2).sum(), np.polyfit(before, after, 1)[0],
                     np.linalg.norm(d) / np.linalg.norm(old)))
    slope = float(np.mean([r[4] for r in rows]))
    save(a.run, f"{a.name}_weights.svg", chart.scatter(
        a.title or "The edit reverses each neuron's push along one direction",
        "One dot per edited neuron: how strongly its output points along the edit's direction, before and after.",
        series, "Before the edit (projection on the direction)", "After", ref=(slope, f"after = {slope:.1f} × before")))
    print("\n| Tensor | Neurons changed | Share of the change in one direction | After ÷ before | Change, relative to those columns |")
    print("|---|---|---|---|---|")
    for label, name, n, share, k, rel in rows:
        print(f"| `{name}` | {n} | {100 * share:.2f}% | {k:.2f} | {100 * rel:.0f}% |")


def answers(d):
    path = os.path.join(d, "answers.jsonl")
    f = gzip.open(path + ".gz", "rt") if os.path.exists(path + ".gz") else open(path)
    return [json.loads(line) for line in f]


def progress(d):
    pts = []
    for line in open(os.path.join(d, "progress.log")):
        m = re.match(r"\s*(\d+) s .*?([\d.]+) tokens/s", line)
        if m:
            t = int(m.group(1)) / 60
            if pts and t < pts[-1][0]:      # the clock restarted: a resumed run; keep only its last session
                pts = []
            pts.append((t, float(m.group(2))))
    return pts


def passes(a):
    ledger = [json.loads(line) for line in open(os.path.join(a.run, "passes.jsonl"))]
    chosen = [next(p for p in ledger if p["pass"] == int(k)) for k in a.passes.split(",")]
    labels = (a.names.split(",") if a.names else
              [f"{p['label'].capitalize()} (pass {p['pass']})" for p in chosen])
    recs = [answers(p["dir"]) for p in chosen]
    sources = sorted({r["source"] for r in recs[0]}, key=lambda s: (s in ORDER and ORDER.index(s), s))
    groups = [SOURCE.get(s, s) for s in sources] + ["All"]
    acc = lambda rs, s: 100 * np.mean([r["correct"] for r in rs if s in (None, r["source"])])  # noqa: E731
    series = [(lab, [acc(rs, s) for s in sources] + [acc(rs, None)]) for lab, rs in zip(labels, recs)]
    for ref in a.ref or []:
        name, vals = ref.split("=")
        v = dict(kv.split(":") for kv in vals.split(","))
        series.append((name, [float(v[s]) if s in v else None for s in sources] + [float(v["all"])]))
    n, cap = len(recs[0]), chosen[0]["max_tokens"]
    note = f"{n} answers per model, same problems, at most {cap:,} new tokens each."
    save(a.run, f"{a.name}_accuracy.svg", chart.columns(a.title or "Share of problems answered correctly", note, groups, series))
    save(a.run, f"{a.name}_outcomes.svg", chart.segmented(
        "What happened to each answer", note,
        [(lab, [sum(r["outcome"] == o for r in rs) for o, _ in OUTCOMES]) for lab, rs in zip(labels, recs)],
        [name for _, name in OUTCOMES], unit=" answers"))
    save(a.run, f"{a.name}_lengths.svg", chart.lines(
        "How long the answers are", "Answers sorted from shortest to longest. A flat top at the cap means cut off.",
        [(lab, [(100 * (i + 1) / len(rs), t) for i, t in enumerate(sorted(r["tokens"] for r in rs))])
         for lab, rs in zip(labels, recs)], "Share of answers, shortest first (%)", "tokens", ylim=(0, cap)))
    speed = [(lab, progress(p["dir"])) for lab, p in zip(labels, chosen)]
    if all(len(pts) > 1 for _, pts in speed):
        save(a.run, f"{a.name}_speed.svg", chart.lines(
            "Generation speed during the pass", "All running answers together. Speed falls as the answers get longer.",
            speed, "Minutes since the pass started", "tokens/s", ylim=(0, max(y for _, pts in speed for _, y in pts))))

    print("\n| | " + " | ".join(labels) + " |\n|---|" + "---|" * len(labels))
    row = lambda name, f: print(f"| {name} | " + " | ".join(f(rs, p) for rs, p in zip(recs, chosen)) + " |")  # noqa: E731
    row("Correct", lambda rs, p: f"{sum(r['correct'] for r in rs)}/{len(rs)} ({acc(rs, None):.1f}%)")
    for s in sources:
        row(f"Correct, {SOURCE.get(s, s)}", lambda rs, p, s=s: f"{acc(rs, s):.1f}%")
    for o, name in OUTCOMES[1:]:
        row(name, lambda rs, p, o=o: f"{sum(r['outcome'] == o for r in rs)}")
    row("Cut-off answers that end in a verbatim loop", lambda rs, p: str(p["loop_end_among_cut"]))
    row("Mean tokens per answer", lambda rs, p: f"{p['mean_tokens']:,}")
    row("Time", lambda rs, p: f"{p['seconds'] / 60:.0f} min")
    row("Peak memory", lambda rs, p: f"{p['peak_gb']:.1f} GB")
    if len(recs) > 1:
        key = lambda r: (r["id"], r["sample"])  # noqa: E731
        first = {key(r): r["correct"] for r in recs[0]}
        d = np.array([float(r["correct"]) - float(first[key(r)]) for r in recs[-1]])
        print(f"\nPaired, {labels[-1]} against {labels[0]}: {100 * d.mean():+.1f} points "
              f"(standard error {100 * d.std(ddof=1) / np.sqrt(len(d)):.1f}); "
              f"{int((d > 0).sum())} problems gained, {int((d < 0).sum())} lost.")


def layers(a):
    j = json.load(open(os.path.join(a.edit, "localise.json")))
    rows = [r for r in j["profile"] if r["layer"] >= 0]
    save(a.run, f"{a.name}_layers.svg", chart.lines(
        a.title or "Where the looping state differs from the reasoning state",
        f"Per layer, over {len(j['cases'])} looping answers: '{j['groups'][0]}' positions minus '{j['groups'][1]}' positions.",
        [("Agreement between answers (cosine)", [(r["layer"], r["agreement"]) for r in rows]),
         ("Size of the difference ÷ size of the state", [(r["layer"], r["size"]) for r in rows])],
        "Layer", "", ylim=(0, 1)))


def probes(a):
    ledger = {json.loads(line)["pass"]: json.loads(line) for line in open(os.path.join(a.run, "passes.jsonl"))}
    sums = [json.load(open(os.path.join(ledger[int(k)]["dir"], "summary.json"))) for k in a.passes.split(",")]
    names = a.names.split(",") if a.names else [s["label"] for s in sums[1:]]
    groups = list(sums[0]["cases"][0]["groups"])
    mean = lambda s, g: float(np.mean([c["groups"][g]["mean_logp"] for c in s["cases"]]))  # noqa: E731
    series = [(n, [mean(sums[0], g) - mean(s, g) for g in groups]) for n, s in zip(names, sums[1:])]
    top = chart.nice(0, max(v for _, vals in series for v in vals) * 1.15)[-1]
    shown = {"productive": "Reasoning text (before the loop)", "repetitive": "Loop text"}
    save(a.run, f"{a.name}_confidence.svg", chart.columns(
        a.title or "How much each edit lowers the model's confidence in the text",
        f"Drop in log-probability per token against the original model, on {len(sums[0]['cases'])} looping answers "
        "the edit was not built from.", [shown.get(g, g) for g in groups], series, ylabel="drop", ymax=top,
        fmt=lambda v: f"{v:.3f}"))
    print("\n| Model | " + " | ".join(shown.get(g, g) for g in groups) + " |\n|---|" + "---|" * len(groups))
    print(f"| Original (pass {sums[0]['pass']}), log-probability per token | " + " | ".join(f"{mean(sums[0], g):.4f}" for g in groups) + " |")
    for n, s in zip(names, sums[1:]):
        print(f"| {n} (pass {s['pass']}), change | " + " | ".join(f"{mean(s, g) - mean(sums[0], g):+.4f}" for g in groups) + " |")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("weights")
    w.add_argument("--base", required=True)
    w.add_argument("--other", required=True)
    p = sub.add_parser("passes")
    p.add_argument("--passes", required=True)
    p.add_argument("--ref", action="append")
    p.add_argument("--names", default=None, help="comma-separated legend names, one per pass")
    la = sub.add_parser("layers")
    la.add_argument("--edit", required=True, help="an edit folder holding localise.json")
    pr = sub.add_parser("probes")
    pr.add_argument("--passes", required=True, help="probe passes of one request; the first is the original model")
    pr.add_argument("--names", default=None)
    for s in (w, p, la, pr):
        s.add_argument("--run", required=True)
        s.add_argument("--name", required=True)
        s.add_argument("--title", default=None)
    a = ap.parse_args()
    {"weights": weights, "passes": passes, "layers": layers, "probes": probes}[a.cmd](a)


if __name__ == "__main__":
    main()
