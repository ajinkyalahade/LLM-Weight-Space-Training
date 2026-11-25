"""How fast a model generates on this machine, under a protocol.

    python -I wsp/speed.py <model dir> <problems .jsonl> [protocol]

1. Batched: the first 8, 32 and 64 problems, 400 new tokens each: total tokens per second and peak memory.
2. Long context: one answer run to 8,000 tokens, tokens per second in each 2,000-token stretch (answers that loop run
   to the token limit, so the speed far into an answer is the one that sets the cost of a pass).
"""
import json
import os
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm  # noqa: E402
from mlx_lm import stream_generate  # noqa: E402
from mlx_lm.generate import BatchGenerator  # noqa: E402


def batched(model, tok, prompts, proto, new_tokens, size):
    gen = BatchGenerator(model, stop_tokens=[[t] for t in tok.eos_token_ids], sampler=lm.sampler(proto),
                         completion_batch_size=size, prefill_batch_size=min(size, 8))
    gen.insert(prompts, [new_tokens] * len(prompts),
               logits_processors=[lm.processors(proto, len(p)) for p in prompts])
    mx.reset_peak_memory()
    t0, n = time.time(), 0
    while rs := gen.next_generated():
        n += sum(1 for r in rs if r.finish_reason != "stop")
    gen.close()
    return n, time.time() - t0, lm.peak_gb()


def main(model_dir, problems, protocol="think35"):
    proto = lm.PROTOCOLS[protocol]
    model, tok = lm.load(model_dir)
    rows = [json.loads(line) for _, line in zip(range(64), open(problems))]
    prompts = [lm.prompt_ids(tok, r["problem"], proto) for r in rows]
    mx.random.seed(0)
    batched(model, tok, prompts[:2], proto, 16, 2)                      # warm-up
    print("batched, 400 new tokens each")
    for size in (8, 32, 64):
        n, s, gb = batched(model, tok, prompts[:size], proto, 400, size)
        print(f"  batch {size:3d}: {n:6d} tokens in {s:6.1f} s = {n / s:7.1f} tokens/s ({n / s / size:5.1f} per answer), "
              f"peak {gb:.2f} GB")
    print("one answer, run to 8,000 tokens")
    mx.reset_peak_memory()
    t0, marks = time.time(), []
    for r in stream_generate(model, tok, prompts[0], max_tokens=8000, sampler=lm.sampler(proto),
                             logits_processors=lm.processors(proto, len(prompts[0]))):
        if r.generation_tokens % 2000 == 0:
            marks.append((r.generation_tokens, time.time()))
        if r.finish_reason:
            print(f"  finished at {r.generation_tokens} tokens ({r.finish_reason})")
    prev = (0, t0)
    for m in marks:
        print(f"  tokens {prev[0]:5d}-{m[0]:5d}: {(m[0] - prev[0]) / (m[1] - prev[1]):6.1f} tokens/s")
        prev = m
    print(f"  peak {lm.peak_gb():.2f} GB")


if __name__ == "__main__":
    main(*sys.argv[1:4])
