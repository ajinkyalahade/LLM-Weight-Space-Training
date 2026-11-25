"""Apply an edit spec to a model's weights. The original files are never changed: the changed columns of each edited
tensor are written to a folder of their own (tensors/<name>.npz: `columns`, `values`), with a manifest (edit.json) that
says exactly what was done and the hashes before and after. wsp/lm.py load(model_dir, edit_dir) runs the edited model.

    python -I wsp/edit.py --base <model dir> --spec <spec.json> --out runs/<name>/edits/<id>

spec.json:
    {"note": "why this edit",
     "edits": [
       {"tensor": "<name>", "op": "remove_direction", "columns": [..], "strength": 6, "direction": "<u.npy>"},
       {"tensor": "<name>", "op": "scale_columns", "columns": [..], "factor": 0}
     ]}

Operations (a column of an output matrix such as mlp.down_proj or self_attn.o_proj is what one neuron, or one
dimension of one head, writes into the residual stream):
  remove_direction   W[:, cols] -= strength * u (u^T W[:, cols]), u a unit vector of the residual stream.
                     strength 1 stops those columns writing along u; above 1 makes them write against it.
  scale_columns      W[:, cols] *= factor. 0 switches the columns off, a negative factor reverses them.
Several edits of one tensor are applied in the order given. Paths in the spec are relative to the spec file.

Refused, as in the post's harness: the token embedding / output layer, the final norm, and any edit that only
multiplies a whole tensor by a constant. Values are stored as the model stores them (bfloat16 here), so the manifest's
"changed" counts are counts of stored values that differ.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import st  # noqa: E402

OUT_OF_SCOPE = ("embed_tokens", "lm_head", "language_model.norm.", "model.norm.")


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def apply(w, e, spec_dir):
    cols = np.asarray(e["columns"], dtype=np.int64)
    assert len(set(cols.tolist())) == len(cols), "a column is listed twice"
    if e["op"] == "remove_direction":
        u = np.load(os.path.join(spec_dir, e["direction"])).astype(np.float64)
        assert u.shape == (w.shape[0],), (u.shape, w.shape)
        u = u / np.linalg.norm(u)
        w[:, cols] -= float(e["strength"]) * np.outer(u, u @ w[:, cols])
    elif e["op"] == "scale_columns":
        w[:, cols] *= float(e["factor"])
    else:
        raise ValueError(f"unknown op {e['op']}")
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    spec_dir = os.path.dirname(os.path.abspath(a.spec))
    base = st.Dir(a.base)

    work, order = {}, []
    for e in spec["edits"]:
        name = e["tensor"]
        assert name in base.of, f"no tensor {name}"
        assert not any(s in name for s in OUT_OF_SCOPE), f"out of scope: {name}"
        assert len(base.info(name)[1]) == 2, f"{name} is not a matrix"
        if name not in work:
            work[name] = base.f32(name).astype(np.float64)
            order.append(name)
        work[name] = apply(work[name], e, spec_dir)

    report, save = [], {}
    for name in order:
        dt, shape = base.info(name)
        old = base.bits(name)
        x = work[name].astype(np.float32)
        new = st.f32_to_bf16(x) if dt == "BF16" else x.astype(st.NP[dt])
        new_f = st.bf16_to_f32(new) if dt == "BF16" else new.astype(np.float32)
        old_f = base.f32(name)
        d = new_f.astype(np.float64) - old_f
        ratio = new_f[old_f != 0] / old_f[old_f != 0]
        # 1% tolerance: a bfloat16 value is only exact to 0.4%, so a constant factor shows up as a ratio that wobbles
        assert not (ratio.size and np.allclose(ratio, np.median(ratio), rtol=1e-2) and (old != new).mean() > 0.9), \
            f"{name}: the edit only multiplies the whole tensor by a constant"
        cols = np.where(np.abs(d).sum(0) > 0)[0]
        save[name] = {"columns": cols.astype(np.int32), "values": new_f[:, cols]}     # only the changed columns
        report.append({"tensor": name, "shape": list(shape), "stored_as": dt,
                       "changed_values": int((old != new).sum()), "changed_columns": int(len(cols)),
                       "relative_change": round(float(np.linalg.norm(d) / np.linalg.norm(old_f)), 5),
                       "sha256_before": sha(old), "sha256_after": sha(new)})
        print(f"{name}: {report[-1]['changed_values']:,} values in {len(cols)} columns changed, "
              f"relative change {report[-1]['relative_change']}")
    os.makedirs(os.path.join(a.out, "tensors"), exist_ok=True)       # nothing is written unless every check passed
    for name, z in save.items():
        np.savez(os.path.join(a.out, "tensors", name + ".npz"), **z)
    json.dump({"base": os.path.abspath(a.base), "spec": spec, "result": report,
               "total_changed_values": sum(r["changed_values"] for r in report)},
              open(os.path.join(a.out, "edit.json"), "w"), indent=1)
    print(f"{len(report)} tensors -> {a.out}")


if __name__ == "__main__":
    main()
