"""Figures and facts for the write-up. Every number is recomputed from the raw answer files under runs/.

    .venv/bin/python -I writeup/figures/make_figures.py [--features <m3_feat_ans.npz>]

Writes figN_*.png (300 dpi, for the paper) and figN_*.svg (for the blog) next to this file, and facts.json with every
paired comparison cited in the text. Paired statistics: per-problem difference (edited − original, each 0/1), mean
and standard error of the mean (ddof 1); z = mean / SE. The pass rule fixed before every sealed test was z >= 2.
"""
import argparse
import gzip
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))
R = lambda run, p: os.path.join(ROOT, "runs", run, "passes", p)  # noqa: E731

# palette: the validated categorical slots used by wsp/chart.py (dataviz reference palette, light mode)
BLUE, ORANGE, GREEN, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY, INK, INK2, GRID = "#a9a8a2", "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({
    "font.family": "Helvetica", "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "svg.fonttype": "none", "hatch.linewidth": 0.8,
})

MODELS = [("Qwen3.5-0.8B", "math"), ("Qwen3.5-2B", "math"), ("MiniCPM5-2B", "code")]
SEALED = {"Qwen3.5-0.8B": ("qwen35-0.8b", "gen_22_holdout_original", "gen_23_holdout_m2"),
          "Qwen3.5-2B": ("qwen35-2b", "gen_10_sealed_original", "gen_11_sealed_m2a"),
          "MiniCPM5-2B": ("minicpm5-2b", "gen_13_sealed_original", "gen_14_sealed_m3c")}


def load(run, p):
    recs = [json.loads(line) for line in gzip.open(os.path.join(R(run, p), "answers.jsonl.gz"))]
    return {(r["id"], r.get("sample", 0)): r for r in recs}


def paired(run, a, b, label, group, kind):
    A, B = load(run, a), load(run, b)
    keys = sorted(A)
    assert set(keys) == set(B), (run, a, b)
    x = np.array([float(B[k]["correct"]) - float(A[k]["correct"]) for k in keys])
    se = x.std(ddof=1) / math.sqrt(len(x))
    return {"label": label, "group": group, "kind": kind, "run": run, "a": a, "b": b, "n": len(x),
            "orig": int(sum(A[k]["correct"] for k in keys)), "edit": int(sum(B[k]["correct"] for k in keys)),
            "diff_pts": round(100 * x.mean(), 1), "se_pts": round(100 * se, 1),
            "z": round(x.mean() / se, 2) if se else 0.0, "gained": int((x > 0).sum()), "lost": int((x < 0).sum())}


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(OUT, name + ".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=None, help="model 3 hidden-state features (closedir measure --answer-region)")
    a = ap.parse_args()
    facts = {"comparisons": [], "sealed": {}}

    # ---- every edit tested against its own original -------------------------------------------------------------
    C = facts["comparisons"]
    C += [paired("qwen35-0.8b", "gen_1_original", "gen_10_published", "Published edit, reproduced (96 problems, 32k cap)", "Qwen3.5-0.8B", "practice"),
          paired("qwen35-0.8b", "gen_16_original" if False else "gen_16_fresh_original", "gen_14_fresh_mine_a", "Edit A: early vs late text", "Qwen3.5-0.8B", "practice"),
          paired("qwen35-0.8b", "gen_16_fresh_original", "gen_15_fresh_published", "Published edit (64 new problems)", "Qwen3.5-0.8B", "practice"),
          paired("qwen35-0.8b", "gen_16_fresh_original", "gen_20_fresh_mine_m2", "Edit M2: matched replays", "Qwen3.5-0.8B", "practice"),
          paired("qwen35-0.8b", "gen_15_fresh_published", "gen_20_fresh_mine_m2", "M2 vs published edit", "Qwen3.5-0.8B", "head-to-head"),
          paired("qwen35-0.8b", "gen_10_published", "gen_21_m2_96", "M2 vs published edit (96, 32k cap)", "Qwen3.5-0.8B", "head-to-head"),
          paired("qwen35-0.8b", "gen_22_holdout_original", "gen_23_holdout_m2", "Edit M2, sealed (256)", "Qwen3.5-0.8B", "sealed"),
          paired("qwen35-0.8b", "gen_16_fresh_original", "gen_26_fresh_m2_4bit", "M2 at 4-bit vs original at full precision", "Qwen3.5-0.8B", "4-bit"),
          paired("qwen35-2b", "gen_1_original", "gen_5_fresh_m2a", "Edit m2a: model 1 recipe", "Qwen3.5-2B", "practice"),
          paired("qwen35-2b", "gen_8_confirm_original", "gen_9_confirm_m2a", "Edit m2a, confirmation (64 new)", "Qwen3.5-2B", "practice"),
          paired("qwen35-2b", "gen_10_sealed_original", "gen_11_sealed_m2a", "Edit m2a, sealed (256)", "Qwen3.5-2B", "sealed"),
          paired("qwen35-2b", "gen_13_fresh_orig_4bit", "gen_12_fresh_4bit", "m2a at 4-bit vs original at 4-bit", "Qwen3.5-2B", "4-bit"),
          paired("minicpm5-2b", "gen_1_original", "gen_7_m3b", "Edit m3b: calibrated on thinking only", "MiniCPM5-2B", "practice"),
          paired("minicpm5-2b", "gen_1_original", "gen_10_m3c", "Edit m3c: calibrated everywhere", "MiniCPM5-2B", "practice"),
          paired("minicpm5-2b", "gen_11_original_s1", "gen_12_m3c_s1", "Edit m3c, new seed", "MiniCPM5-2B", "practice"),
          paired("minicpm5-2b", "gen_13_sealed_original", "gen_14_sealed_m3c", "Edit m3c, sealed (128)", "MiniCPM5-2B", "sealed"),
          paired("minicpm5-2b", "gen_17_orig_4bit", "gen_18_m3c_4bit", "m3c at 4-bit vs original at 4-bit", "MiniCPM5-2B", "4-bit")]

    # ---- sealed tests: per-model outcome breakdown -------------------------------------------------------------
    for m, (run, po, pe) in SEALED.items():
        d = {}
        for tag, p in (("original", po), ("edited", pe)):
            rs = list(load(run, p).values())
            n = len(rs)
            d[tag] = {"n": n, "correct": sum(r["correct"] for r in rs),
                      "wrong": sum(r["outcome"] == "wrong_answer" for r in rs),
                      "no_answer": sum(r["outcome"] == "no_answer_finished" for r in rs),
                      "cut_loop": sum(r["outcome"] == "cut_at_max_tokens" and r.get("loop_end", False) for r in rs),
                      "cut_other": sum(r["outcome"] == "cut_at_max_tokens" and not r.get("loop_end", False) for r in rs),
                      "closed_thinking": sum(bool(r.get("closed_thinking")) for r in rs),
                      "mean_tokens": round(float(np.mean([r["tokens"] for r in rs]))),
                      "tokens": sorted(r["tokens"] for r in rs),
                      "by_source": {s: [sum(r["correct"] for r in rs if r["source"] == s), sum(r["source"] == s for r in rs)]
                                    for s in sorted({r["source"] for r in rs})}}
        facts["sealed"][m] = d

    # ---- Figure 1: sealed accuracy -------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    xs = np.arange(3)
    for j, (tag, col, name) in enumerate((("original", GRAY, "Original"), ("edited", BLUE, "Edited (locked edit)"))):
        vals, lo, hi = [], [], []
        for m, _ in MODELS:
            s = facts["sealed"][m][tag]
            v = 100 * s["correct"] / s["n"]
            w = wilson(s["correct"], s["n"])
            vals.append(v); lo.append(v - w[0]); hi.append(w[1] - v)
        b = ax.bar(xs + (j - 0.5) * 0.34, vals, 0.32, color=col, label=name, yerr=[lo, hi],
                   error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2})
        for rect, v, h in zip(b, vals, hi):
            ax.text(rect.get_x() + rect.get_width() / 2, v + h + 2, f"{v:.1f}%", ha="center", fontsize=8, color=INK)
    ax.set_xticks(xs, [f"{m}\n({t}, n = {facts['sealed'][m]['original']['n']})" for m, t in MODELS])
    ax.set_ylim(0, 100); ax.set_ylabel("Problems solved (%)"); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=2)
    save(fig, "fig1_sealed_accuracy")

    # ---- Figure 2: forest plot, every comparison ------------------------------------------------------------------
    order = [c for c in C if c["kind"] != "head-to-head"] + [c for c in C if c["kind"] == "head-to-head"]
    groups = []
    for c in order:
        grp = c["group"] if c["kind"] != "head-to-head" else "Head-to-head (M2 minus published edit)"
        if not groups or groups[-1] != grp:
            groups.append(grp)
    nrows = len(order) + len(groups)
    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    fig.subplots_adjust(left=0.43, right=0.80)
    ty = ax.get_yaxis_transform()
    y = nrows - 1
    prev = None
    for c in order:
        grp = c["group"] if c["kind"] != "head-to-head" else "Head-to-head (M2 minus published edit)"
        if grp != prev:
            ax.text(-0.75, y, grp, transform=ty, fontsize=8.5, fontweight="bold", color=INK, ha="left", va="center")
            prev = grp
            y -= 1
        z = c["z"]
        col, fill = (BLUE, BLUE) if z >= 2 else ((ORANGE, ORANGE) if z <= -2 else (INK2, "white"))
        ax.errorbar(c["diff_pts"], y, xerr=2 * c["se_pts"], fmt="s" if c["kind"] == "sealed" else "o", color=col,
                    mfc=fill, mec=col, ms=5.5, elinewidth=1.2, capsize=0)
        ax.text(-0.72, y, c["label"], transform=ty, va="center", fontsize=7.6, color=INK if c["kind"] == "sealed" else INK2,
                fontweight="bold" if c["kind"] == "sealed" else "normal")
        ax.text(1.03, y, f"{c['diff_pts']:+.1f}  (z {z:+.1f}, {c['gained']}/{c['lost']})", transform=ty,
                va="center", fontsize=7.4, color=INK2)
        y -= 1
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(-30, 55); ax.set_ylim(-0.8, nrows + 0.2)
    ax.set_xticks([-20, 0, 20, 40]); ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.text(1.03, nrows - 1, "diff (z, gained/lost)", transform=ty, fontsize=7.4, color=INK2, fontweight="bold", va="center")
    ax.set_xlabel("Paired difference, points (edited − original); bars = ±2 SE")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="better (z ≥ 2)"),
                       Line2D([], [], marker="o", ls="", color=INK2, mfc="white", label="not established"),
                       Line2D([], [], marker="o", ls="", color=ORANGE, label="worse (z ≤ −2)"),
                       Line2D([], [], marker="s", ls="", color=INK2, label="sealed test (bold)")],
              loc="upper center", fontsize=7.4, bbox_to_anchor=(0.5, -0.09), ncol=4)
    save(fig, "fig2_forest")

    # ---- Figure 3: what happened to each answer, sealed ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    parts = [("correct", "Correct", BLUE, None), ("wrong", "Wrong answer", ORANGE, None),
             ("no_answer", "Finished, no answer", GREEN, None),
             ("cut_loop", "Cut at cap, verbatim loop", YELLOW, "////"), ("cut_other", "Cut at cap, no loop", YELLOW, None)]
    rows = []
    for m, _ in MODELS:
        rows += [(f"{m} original", facts["sealed"][m]["original"]), (f"{m} edited", facts["sealed"][m]["edited"])]
    for i, (name, s) in enumerate(rows[::-1]):
        left = 0
        for key, lab, col, hatch in parts:
            w = 100 * s[key] / s["n"]
            if w:
                ax.barh(i, w, left=left, color=col, edgecolor="white", hatch=hatch, linewidth=1.2,
                        label=lab if i == len(rows) - 1 or key not in [p for p, *_ in parts[:0]] else None)
                if w >= 7:
                    ax.text(left + 1.2, i, f"{w:.0f}%", va="center", fontsize=7, color="white" if col != YELLOW else INK)
            left += w
    ax.set_yticks(range(len(rows)), [n for n, _ in rows[::-1]], fontsize=7.6)
    ax.set_xlim(0, 100); ax.set_xlabel("Share of sealed-test answers (%)"); ax.grid(False)
    h, l = ax.get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h, l):
        seen.setdefault(ll, hh)
    ax.legend([seen[p[1]] for p in parts if p[1] in seen], [p[1] for p in parts if p[1] in seen],
              loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=3, fontsize=7.4)
    save(fig, "fig3_outcomes")

    # ---- Figure 4: answer lengths, sealed --------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.3), sharey=True)
    for ax, (m, _) in zip(axs, MODELS):
        for tag, col, name in (("original", GRAY, "Original"), ("edited", BLUE, "Edited")):
            t = facts["sealed"][m][tag]["tokens"]
            ax.plot(100 * (np.arange(len(t)) + 1) / len(t), t, color=col, lw=1.8, label=name)
        ax.set_title(m, fontsize=8.5, color=INK); ax.set_xlabel("Answers, shortest first (%)", fontsize=7.6)
        ax.set_ylim(0, 17000); ax.set_xlim(0, 100)
    axs[0].set_ylabel("Tokens in answer"); axs[0].legend(loc="upper left", fontsize=7.4)
    fig.tight_layout()
    save(fig, "fig4_lengths")

    # ---- Figure 5: model 3 mechanism -------------------------------------------------------------------------------
    if a.features:
        z = np.load(a.features)
        H, pos, ans, closing, lineend = z["H"], z["pos"], z["answer"].astype(bool), z["closing"].astype(bool), z["lineend"].astype(bool)
        delta = np.load(os.path.join(ROOT, "runs/minicpm5-2b/edits/m3c/d.npy")).astype(np.float32)
        boost = np.concatenate([H[i:i + 20000].astype(np.float32) @ delta for i in range(0, len(H), 20000)])
        think = ~ans
        bins = np.arange(0, 16385, 1024)
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        stats = {}
        for mask, col, name in ((think & lineend & ~closing, BLUE, "Thinking, end of a line"),
                                (think & ~lineend & ~closing, ORANGE, "Thinking, inside a line")):
            mid, med, q1, q3 = [], [], [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                v = boost[mask & (pos >= lo) & (pos < hi)]
                if len(v) >= 30:
                    mid.append((lo + hi) / 2); med.append(np.median(v)); q1.append(np.percentile(v, 10)); q3.append(np.percentile(v, 90))
            ax.plot(mid, med, color=col, lw=1.8, label=name + " (median, 10–90th pct)")
            ax.fill_between(mid, q1, q3, color=col, alpha=0.15, lw=0)
            stats[name] = {"bin_mid": mid, "median": [round(float(v), 2) for v in med]}
        va = boost[ans]
        ax.axhspan(np.percentile(va, 10), np.percentile(va, 90), color=GREEN, alpha=0.15, lw=0)
        ax.axhline(np.median(va), color=GREEN, lw=1.5, ls="--", label="Answer section, any position (median, 10–90th)")
        cl = closing & think
        ax.scatter(pos[cl], boost[cl], s=10, color=INK, zorder=3, label="Where the model really closed thinking")
        ax.set_xlabel("Position in the thinking (tokens)"); ax.set_ylabel("Added to the </think> logit")
        ax.set_xlim(0, 16384); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7, ncol=2)
        save(fig, "fig5_mechanism")
        facts["mechanism"] = {"positions": int(len(H)), "answer_median": round(float(np.median(va)), 2),
                              "answer_p90": round(float(np.percentile(va, 90)), 2),
                              "closing_median": round(float(np.median(boost[cl])), 2), "closing_n": int(cl.sum()),
                              "by_position": stats,
                              "corr_lineend_pos": round(float(np.corrcoef(pos[think & lineend], boost[think & lineend])[0, 1]), 3)}

    # ---- Figure 6: 4-bit ---------------------------------------------------------------------------------------------
    q = {"Qwen3.5-2B": ("qwen35-2b", "gen_1_original", "gen_5_fresh_m2a", "gen_13_fresh_orig_4bit", "gen_12_fresh_4bit"),
         "MiniCPM5-2B": ("minicpm5-2b", "gen_1_original", "gen_10_m3c", "gen_17_orig_4bit", "gen_18_m3c_4bit")}
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    labels = ["Original\nfull precision", "Edited\nfull precision", "Original\n4-bit", "Edited\n4-bit"]
    facts["quant"] = {}
    for j, (m, (run, *ps)) in enumerate(q.items()):
        vals = []
        for p in ps:
            rs = list(load(run, p).values())
            vals.append(100 * sum(r["correct"] for r in rs) / len(rs))
        facts["quant"][m] = [round(v, 1) for v in vals]
        xs = np.arange(4) * 0.2 + j * 1.1
        for k, (x, v) in enumerate(zip(xs, vals)):
            ax.bar(x, v, 0.18, color=BLUE if k % 2 else GRAY, hatch="////" if k >= 2 else None, edgecolor="white", lw=1)
            ax.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=7.6, color=INK)
        ax.text(xs.mean(), -6, m, ha="center", fontsize=8.5, color=INK)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GRAY, label="Original"), Patch(color=BLUE, label="Edited"),
                       Patch(facecolor="white", edgecolor=INK2, hatch="////", label="4-bit (MLX, 4.5 bits/weight)")],
              loc="upper right", fontsize=7.4)
    ax.set_xticks([]); ax.set_ylim(0, 80); ax.set_ylabel("Practice problems solved (%, n = 64)"); ax.grid(axis="x", visible=False)
    save(fig, "fig6_quantization")

    for m in facts["sealed"]:
        for tag in ("original", "edited"):
            facts["sealed"][m][tag].pop("tokens")
    json.dump(facts, open(os.path.join(OUT, "facts.json"), "w"), indent=1)
    for c in C:
        print(f"{c['group']:13s} {c['kind']:12s} {c['label'][:52]:52s} {c['orig']:3d}->{c['edit']:3d}/{c['n']:3d} "
              f"{c['diff_pts']:+6.1f} SE {c['se_pts']:4.1f} z {c['z']:+5.2f} {c['gained']}/{c['lost']}")
    print(json.dumps({k: v for k, v in facts.items() if k != "comparisons"}, indent=1)[:3500])


if __name__ == "__main__":
    main()
