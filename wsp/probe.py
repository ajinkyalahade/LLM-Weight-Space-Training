"""A teacher-forced pass: the model reads a fixed text (it generates nothing) and what happens inside is recorded.

As a library:
    out = probe.forward(model, ids, layers=[15, 19])
        out["logp"][t]     log-probability the model gave to token t, given the tokens before it (t = 0 is nan)
        out["resid"][L]    residual stream after layer L, (T, width), for every layer; L = -1 is the embedding
        out["act"][L]      MLP neuron activations of layer L, (T, neurons): the input of mlp.down_proj, so
                           neuron n adds act[t, n] * down_proj.weight[:, n] to the residual stream at position t
    All arrays are float32 numpy.

As a command (counts as one forward pass in runs/<name>/passes.jsonl):
    python -I wsp/probe.py --run runs/<name> --model <dir> [--edit <edit dir>] --request <request.json> --label <name>
    python -I wsp/probe.py --selftest --model <dir>

request.json:
    {"layers": [15, 19],
     "cases": [{"name": "loop_3", "tokens": [..token ids, prompt included..],
                "groups": {"productive": [[120, 480]], "repetitive": [[2000, 2600]]}}]}
A group is a set of ranges [first, last) of TOKENS BEING PREDICTED. For token t the state that produced it is the one
at position t - 1, so that is the position whose residual stream and activations are averaged. Written to
runs/<name>/passes/fwd_<k>_<label>/: summary.json (per case and group: tokens, mean log-probability) and means.npz
(`resid/<case>/<group>`: (layers + 1, width), row 0 the embedding; `act/<case>/<group>/<layer>`: (neurons,)).
Only these averages are saved: raw activations for one 8,000-token text would be about 300 MB.

Written for the architecture mlx-lm calls qwen3_5 (pre-norm layers with `linear_attn` or `self_attn`, and an MLP with
gate_proj / up_proj / down_proj). --selftest checks that this file's layer-by-layer pass matches the model's own.
"""
import argparse
import gzip
import json
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm  # noqa: E402
from mlx_lm.models.base import create_attention_mask, create_ssm_mask  # noqa: E402


def _np(a):
    return np.array(a.astype(mx.float32))


def forward(model, ids, layers=(), want_resid=True, chunk=512):
    tm = model.model
    h = tm.embed_tokens(mx.array([list(ids)]))
    fa, ssm = create_attention_mask(h, None), create_ssm_mask(h, None)
    out = {"resid": {-1: _np(h[0])} if want_resid else {}, "act": {}}
    for i, layer in enumerate(tm.layers):
        a_in = layer.input_layernorm(h)
        r = layer.linear_attn(a_in, ssm, None) if layer.is_linear else layer.self_attn(a_in, fa, None)
        mid = h + r
        z = layer.post_attention_layernorm(mid)
        act = nn.silu(layer.mlp.gate_proj(z)) * layer.mlp.up_proj(z)
        h = mid + layer.mlp.down_proj(act)
        mx.eval(h)
        if want_resid:
            out["resid"][i] = _np(h[0])
        if i in layers:
            out["act"][i] = _np(act[0])
    final = tm.norm(h)[0]
    head = tm.embed_tokens.as_linear if model.language_model.args.tie_word_embeddings else model.language_model.lm_head
    target = mx.array(list(ids)[1:])
    logp = [np.array([np.nan], np.float32)]
    for s in range(0, len(ids) - 1, chunk):                 # the output layer in pieces: all of it at once is GBs
        lg = head(final[s:s + chunk]).astype(mx.float32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        t = target[s:s + chunk]
        logp.append(np.array(mx.take_along_axis(lp[:t.shape[0]], t[:, None], axis=-1)[:, 0]))
    out["logp"] = np.concatenate(logp)
    out["final"] = final
    return out


def positions(ranges, n):
    p = np.concatenate([np.arange(max(1, a), min(n, b)) for a, b in ranges]) if ranges else np.array([], int)
    return np.unique(p)


def selftest(model, tok):
    ids = lm.prompt_ids(tok, "What is 17 * 23?", lm.PROTOCOLS["think35"])
    ids = ids + tok.encode("We need to multiply 17 by 23. 17 * 20 = 340 and 17 * 3 = 51, so the total is 391.")
    out = forward(model, ids, layers=[0])
    ref = model(mx.array([ids]))[0].astype(mx.float32)
    ref = ref - mx.logsumexp(ref, axis=-1, keepdims=True)
    ref_lp = np.array(mx.take_along_axis(ref[:-1], mx.array(ids[1:])[:, None], axis=-1)[:, 0])
    diff = np.abs(out["logp"][1:] - ref_lp).max()
    w = np.array(model.model.layers[0].mlp.down_proj.weight.astype(mx.float32))
    print(f"{len(ids)} tokens; largest difference in log-probability between this pass and the model's own: {diff:.2e}")
    print(f"residual stream: {len(out['resid'])} snapshots of shape {out['resid'][0].shape}; "
          f"layer-0 activations {out['act'][0].shape}; down_proj {w.shape}")
    assert diff < 1e-3, "this file's forward pass does not match the model's"
    print("selftest passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--model", required=True)
    ap.add_argument("--edit", default=None)
    ap.add_argument("--request")
    ap.add_argument("--label", default="probe")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    model, tok = lm.load(a.model, a.edit)
    if a.selftest:
        return selftest(model, tok)

    req = json.load(gzip.open(a.request, "rt") if a.request.endswith(".gz") else open(a.request))
    ledger = os.path.join(a.run, "passes.jsonl")
    k = lm.next_pass(a.run, "fwd", a.label)
    out_dir = os.path.join(a.run, "passes", f"fwd_{k}_{a.label}")
    os.makedirs(out_dir, exist_ok=True)
    t0, means, cases, total = time.time(), {}, [], 0
    for case in req["cases"]:
        ids = case["tokens"]
        out = forward(model, ids, layers=req.get("layers", ()))
        total += len(ids)
        row = {"name": case["name"], "tokens": len(ids), "mean_logp": float(np.nanmean(out["logp"])), "groups": {}}
        for g, ranges in case.get("groups", {}).items():
            p = positions(ranges, len(ids))
            row["groups"][g] = {"tokens": int(len(p)), "mean_logp": float(out["logp"][p].mean()) if len(p) else None}
            if len(p):
                means[f"resid/{case['name']}/{g}"] = np.stack([out["resid"][i][p - 1].mean(0)
                                                             for i in sorted(out["resid"])])
                for layer, act in out["act"].items():
                    means[f"act/{case['name']}/{g}/{layer}"] = act[p - 1].mean(0)
        np.save(os.path.join(out_dir, f"logp_{case['name']}.npy"), out["logp"].astype(np.float16))
        cases.append(row)
        mx.clear_cache()
    np.savez_compressed(os.path.join(out_dir, "means.npz"), **means)
    with gzip.open(os.path.join(out_dir, "request.json.gz"), "wt") as dst:
        json.dump(req, dst)                                # the pass folder holds everything needed to repeat it
    summary = {"pass": k, "type": "forward", "label": a.label, "model": a.model, "edit": a.edit,
               "request": "request.json.gz", "layers": req.get("layers", []), "tokens": total, "cases": cases,
               "seconds": round(time.time() - t0, 1), "peak_gb": round(lm.peak_gb(), 2), "dir": out_dir}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    with open(ledger, "a") as f:
        f.write(json.dumps({**{x: summary[x] for x in summary if x != "cases"},
                            "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
