"""From a probe pass to an edit: the direction that separates two groups of positions, and the neurons that write it.

    python -I wsp/localise.py --pass-dir runs/<name>/passes/fwd_<k>_<label> --base <model dir> --layers 15 19 \
        [--a repetitive] [--b productive] [--top 64] [--strength 6] [--out runs/<name>/edits/<id>] \
        [--ref-u "runs/<name>/published_u{}.npy"] [--ref-edit runs/<name>/edits/published_rebuild/edit.json]

For every case of the pass that has both groups, d = mean residual stream in group a - mean in group b, per layer.
    all layers   how large d is next to the residual stream itself, and how alike the cases' d are (mean cosine
                 between pairs of cases): a layer where both are high is a candidate for an edit
    per layer    u = the mean d over cases, made unit length
    writers      for each MLP neuron n of the layer:  write[n] = (mean activation in group a) * (u . column n of
                 down_proj), i.e. how far the neuron moves the residual stream along u at an average group-a position
With --ref-u / --ref-edit, each result is compared with a known edit (cosine of directions, neurons in common).
With --matched, the pass is a probe of a request from wsp/branch.py: d is computed per prefix, looping continuations
minus non-looping continuations of the same text.
With --out, writes u_<layer>.npy and spec.json there: `remove_direction` on the --top writers of each layer at
--strength, ready for wsp/edit.py. Nothing is applied here.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import st  # noqa: E402

TENSOR = "model.language_model.layers.{}.mlp.down_proj.weight"


def cos(x, y):
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass-dir", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--a", default="repetitive")
    ap.add_argument("--b", default="productive")
    ap.add_argument("--top", type=int, default=64)
    ap.add_argument("--strength", type=float, default=6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref-u", default=None)
    ap.add_argument("--ref-edit", default=None)
    ap.add_argument("--tensor", default=TENSOR)
    ap.add_argument("--matched", action="store_true", help="the pass is a matched-pairs probe from wsp/branch.py")
    a = ap.parse_args()

    z = np.load(os.path.join(a.pass_dir, "means.npz"))
    names = sorted({k.split("/")[1] for k in z.files if k.startswith("resid/")})
    if a.matched:
        # cases "<prefix>__<a|b>_<i>" with one group "cont": per prefix, mean over its a-continuations minus mean
        # over its b-continuations, so both sides start from the identical text
        a.a, a.b = (a.a, a.b) if a.a != "repetitive" else ("rep", "prod")
        by = {}
        for n in names:
            prefix, rest = n.rsplit("__", 1)
            by.setdefault(prefix, {}).setdefault(rest.split("_")[0], []).append(n)
        cases = sorted(p for p, g in by.items() if a.a in g and a.b in g)
        mean = lambda key: lambda p, side: np.mean([z[key(n)] for n in by[p][side]], axis=0)  # noqa: E731
        res, act = mean(lambda n: f"resid/{n}/cont"), {}
        A = np.stack([res(p, a.a) for p in cases]).astype(np.float64)
        B = np.stack([res(p, a.b) for p in cases]).astype(np.float64)
        act_of = lambda layer, side: np.mean([mean(lambda n: f"act/{n}/cont/{layer}")(p, side)  # noqa: E731
                                              for p in cases], axis=0).astype(np.float64)
    else:
        cases = [c for c in names if f"resid/{c}/{a.a}" in z.files and f"resid/{c}/{a.b}" in z.files]
        A = np.stack([z[f"resid/{c}/{a.a}"] for c in cases]).astype(np.float64)   # (cases, layers + 1, width)
        B = np.stack([z[f"resid/{c}/{a.b}"] for c in cases]).astype(np.float64)
        act_of = lambda layer, side: np.mean([z[f"act/{c}/{side}/{layer}"] for c in cases],  # noqa: E731
                                             axis=0).astype(np.float64)
    assert cases, "no case has both groups"
    D = A - B
    print(f"{len(cases)} {'prefixes' if a.matched else 'cases'}; d = '{a.a}' - '{a.b}'\n")
    print("layer   |d| / |residual|   agreement between cases" + ("   cosine with reference" if a.ref_u else ""))
    profile = []
    for row in range(D.shape[1]):
        layer = row - 1
        rel = np.linalg.norm(D[:, row].mean(0)) / np.linalg.norm(B[:, row], axis=1).mean()
        pairs = [cos(D[i, row], D[j, row]) for i in range(len(cases)) for j in range(i)]
        ref = ""
        if a.ref_u and os.path.exists(a.ref_u.format(layer)):
            ref = f"{cos(D[:, row].mean(0), np.load(a.ref_u.format(layer))):22.3f}"
        print(f"{'embed' if layer < 0 else layer:>5}   {rel:16.3f}   {np.mean(pairs) if pairs else float('nan'):23.3f}{ref}")
        profile.append({"layer": layer, "size": round(float(rel), 4),
                        "agreement": round(float(np.mean(pairs)), 4) if pairs else None})

    base = st.Dir(a.base)
    ref_cols = {}
    if a.ref_edit:
        e = json.load(open(a.ref_edit))
        for item in (e.get("spec") or e)["edits"]:
            ref_cols[item["tensor"]] = set(item["columns"])
    spec = {"note": f"'{a.a}' minus '{a.b}' direction from {a.pass_dir}; top {a.top} writers per layer; "
                    f"strength {a.strength}", "edits": []}
    found = {"pass_dir": a.pass_dir, "cases": cases, "groups": [a.a, a.b], "profile": profile, "layers": {}}
    for layer in a.layers:
        name = a.tensor.format(layer)
        u = D[:, layer + 1].mean(0)
        u /= np.linalg.norm(u)
        W = base.f32(name).astype(np.float64)
        proj = u @ W
        act_a, act_b = act_of(layer, a.a), act_of(layer, a.b)
        write = act_a * proj
        order = np.argsort(-write)
        top = order[:a.top]
        print(f"\nlayer {layer}: all {len(write)} neurons together move the residual stream {write.sum():+.3f} along u "
              f"at an average '{a.a}' position ({(act_b * proj).sum():+.3f} at a '{a.b}' one); "
              f"the top {a.top} writers give {write[top].sum():+.3f}")
        print("   top 16 writers: " + ", ".join(f"{n} ({write[n]:+.3f})" for n in order[:16]))
        info = {"write_all": round(float(write.sum()), 4), "write_top": round(float(write[top].sum()), 4),
                "write_all_b": round(float((act_b * proj).sum()), 4),
                "top_writers": [[int(n), round(float(write[n]), 4)] for n in order[:a.top]]}
        if a.ref_u and os.path.exists(a.ref_u.format(layer)):
            info["cosine_with_reference"] = round(cos(u, np.load(a.ref_u.format(layer))), 4)
            print(f"   cosine of u with the reference direction: {info['cosine_with_reference']:.3f}")
        if name in ref_cols:
            rc = ref_cols[name]
            rank = {int(n): i for i, n in enumerate(order)}
            info["in_common_with_reference"] = len(rc & set(top.tolist()))
            info["in_common_top16"] = len(rc & set(order[:16].tolist()))
            print(f"   neurons in common with the reference edit: {info['in_common_with_reference']} of {len(rc)} "
                  f"({info['in_common_top16']} of my top 16); "
                  f"the reference neurons' ranks here: median {int(np.median([rank[c] for c in rc]))}, "
                  f"worst {max(rank[c] for c in rc)}")
        found["layers"][str(layer)] = info
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            np.save(os.path.join(a.out, f"u_{layer}.npy"), u)
            spec["edits"].append({"tensor": name, "op": "remove_direction", "columns": sorted(int(n) for n in top),
                                  "strength": a.strength, "direction": f"u_{layer}.npy"})
    if a.out:
        json.dump(spec, open(os.path.join(a.out, "spec.json"), "w"), indent=1)
        json.dump(found, open(os.path.join(a.out, "localise.json"), "w"), indent=1)
        print(f"\nwrote {a.out}/spec.json (not applied; apply with wsp/edit.py)")


if __name__ == "__main__":
    main()
