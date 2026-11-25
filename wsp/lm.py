"""Loading a model in mlx-lm and generating with a fixed, named protocol.

A protocol is the set of things that must stay the same between the original and the edited model: the chat template
options, the instruction appended to the problem, and the sampling. `think35` is the one the post evaluated
Qwen3.5-0.8B with (reference/lm_understand-logs/exp8_math/README.md, "Evaluation").
"""
import mlx.core as mx
from mlx_lm import load as _load
from mlx_lm.sample_utils import make_sampler

PROTOCOLS = {
    "think35": {
        "enable_thinking": True,
        "suffix": "\nPlease reason step by step, and put your final answer within \\boxed{}.",
        "temp": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
        "max_tokens": 32768,
    },
    "minicpm_code": {                     # MiniCPM5-2B model card: temperature 1.0, top-p 0.95, min-p 0; thinking on
        "enable_thinking": True,
        "suffix": "",                     # the LiveCodeBench prompt is already in the problem text (wsp/lcb.py)
        "temp": 1.0, "top_p": 0.95, "top_k": 0, "presence_penalty": 0,
        "max_tokens": 16384,
    },
}


def load(model_dir, edit_dir=None):
    """(model, tokenizer) from a folder holding config.json, the tokenizer files and .safetensors weights.
    With edit_dir (a folder written by wsp/edit.py), the edited tensors replace the originals in memory: the model
    folder is not changed and nothing is copied."""
    model, tok = _load(model_dir)
    if edit_dir:
        import glob
        import os
        import sys

        import numpy as np
        from mlx.utils import tree_flatten
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import st
        have = dict(tree_flatten(model.parameters()))
        base, new = st.Dir(model_dir), {}
        for p in glob.glob(os.path.join(edit_dir, "tensors", "*.npz")):
            name, z = os.path.basename(p)[:-4], np.load(p)
            w = base.f32(name).copy()
            if "rows" in z:                           # a row edit (wsp/closedir.py: one output-token row of lm_head)
                w[z["rows"]] = z["values"]
            else:
                w[:, z["columns"]] = z["values"]
            new[name] = mx.array(w)
        assert new, f"no edited tensors in {edit_dir}"
        new = model.sanitize(new)                     # file names -> this runtime's parameter names
        for k, v in new.items():
            assert k in have and have[k].shape == v.shape, f"edited tensor {k} does not fit the model"
            new[k] = v.astype(have[k].dtype)
        model.load_weights(list(new.items()), strict=False)
        mx.eval(model.parameters())
    return model, tok


def next_pass(run, kind, label):
    """The number of a new pass in a run folder: one more than every number already used, in the ledger or by a pass
    folder (a pass that is still running has a folder but no ledger line yet). An interrupted pass of the same kind
    and label keeps its number, so running its command again continues it."""
    import json
    import os
    import re
    used = set()
    ledger = os.path.join(run, "passes.jsonl")
    if os.path.exists(ledger):
        used |= {json.loads(line)["pass"] for line in open(ledger)}
    folder = os.path.join(run, "passes")
    for name in os.listdir(folder) if os.path.isdir(folder) else []:
        m = re.match(r"(gen|fwd)_(\d+)_(.+)$", name)
        if not m or name.endswith("_trial"):
            continue
        if (m.group(1), m.group(3)) == (kind, label) and not os.path.exists(os.path.join(folder, name, "summary.json")):
            return int(m.group(2))
        used.add(int(m.group(2)))
    return max(used, default=0) + 1


def prompt_ids(tok, problem, proto):
    """Token ids of the chat prompt for one problem under a protocol."""
    messages = [{"role": "user", "content": problem + proto["suffix"]}]
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                   enable_thinking=proto["enable_thinking"])


def sampler(proto):
    return make_sampler(temp=proto["temp"], top_p=proto["top_p"], top_k=proto["top_k"])


def presence_penalty(penalty, prompt_len):
    """Subtract `penalty` from the logit of every token generated so far (each distinct token once), and of no
    prompt token. mlx-lm's own make_presence_penalty looks only at the last 20 tokens and counts the prompt."""

    state = {"seen": prompt_len, "mask": None}      # a running mask over the vocabulary: cost does not grow with length

    def processor(tokens, logits):
        if state["mask"] is None:
            state["mask"] = mx.zeros(logits.shape[-1], dtype=logits.dtype)
        n = tokens.shape[0]
        if n > state["seen"]:
            state["mask"][tokens[state["seen"]:n]] = 1
            state["seen"] = n
        return logits - penalty * state["mask"]

    return processor


def processors(proto, prompt_len):
    p = proto.get("presence_penalty") or 0
    return [presence_penalty(p, prompt_len)] if p else []


def peak_gb():
    return mx.get_peak_memory() / 1e9
