# Qwen3.5-2B — lab notebook

## Fixed facts

- Model: `Qwen/Qwen3.5-2B`, downloaded 2026-11-16 to `models/Qwen3.5-2B` (4.3 GB, one safetensors file, bf16).
- 24 layers, width 2048, 6,144 MLP neurons per layer, `qwen3_5`, tied embeddings.
- `wsp/probe.py --selftest`: passed (difference 0.00e+00).
- Protocol: `think35` (same as model 1: thinking on, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5).

## Limits (end goal agreed 2026-11-16, top of `plan.md`)

- Started 2026-11-16 ~06:00. **Time limit: 2026-11-18 ~06:00.**
- At most 3 edit attempts of my own (no published fix exists for this model).
- 50 passes. Sealed test once.

## Plan

1. Baseline on the 64 practice problems model 1 used (`runs/qwen35-0.8b/problems_fresh64.json`), cap 16,384, seed 0.
   Compare with model 1's original (12.5%, 54 of 64 answers ending in a loop).
2. If loops dominate: model 1's M2 recipe (loops → rewind and replay → matched direction → top 64 writers, strength
   6) in the corresponding layers. If not: look at what does fail and pick a target, or record "no fix found".

## Log

- **2026-11-16 ~06:05** Downloaded, self-test passed. Baseline (pass 1) started.
- **2026-11-16 (pass 1, 32 min)** Baseline on fresh64, cap 16,384: **36/64 (56.3%)** (GSM8K 62.5%, MATH 50.0%),
  1 wrong, 27 cut (9 end in a verbatim loop), mean 10,459 tokens, peak 7.4 GB. 0.8B original on the same: 8/64.
  Finished answers are almost all right (36/37): the failures are not finishing. Non-loop cut answers are near-loops:
  4-word repeat share in the last 6,000 chars 0.19 vs 0.08 for correct answers; "Wait" 14.6 vs 2.3 per 1,000 words.
  **Target: stuck thinking. Attempt 1 = model 1's M2 recipe, layers 15 and 19.** Only 9 verbatim loops, so a
  collection pass (pass 2) on `problems_collect160.json` (random.Random(3), 80 + 80, excluding the 96 and fresh64).
- **2026-11-16 22:50 (pass 2, 77 min)** Collection: 96/160 (60.0%; GSM8K 71.3%, MATH 48.8%), 2 wrong, 62 cut (15 end
  in a verbatim loop), mean 9,712 tokens, 338 tok/s, peak 7.9 GB.
- `loops.py` (8-gram ≥ 90% over 200 tokens, ≥ 400 after) on passes 1 + 2, `--max-tokens 17000` so late onsets count:
  **25 looping answers** (9 + 16), 304,464 tokens, onset median ~11k (earliest 4,494). Model 1 had 40 for M2.
  Thinner input; recipe kept unchanged (offsets −900 −600 −300, n 8, length 300) → pass 3 `branch`.
- **2026-11-16 23:38 (pass 3, 47 min, peak 10.5 GB)** 600 continuations: 111 rep / 218 prod / rest unclear;
  **all 25 prefixes have both** ⇒ 94 matched cases (model 1: 18 prefixes, 80 cases). Long prefixes (~11k) make it slow.
  Chain started: probe (pass 4) → `localise --matched --layers 15 19 --top 64 --strength 6` → `edits/m2a` →
  generation test on fresh64 (pass 5).
- **Pass 4 `fwd_4_matched`** probe → `edits/m2a` (top 64 writers in L15 and L19, strength 6). One neuron dominates
  each layer: L15 #6069 (+0.317 of +0.845 from the top 64), L19 #5108 (+0.333 of +1.473); relative column change
  0.059 (L15), 0.074 (L19). Same pattern as model 1 (L19 #11).
- **2026-11-17 00:15 (pass 5 `gen_5_fresh_m2a`, 28 min, peak 7.0 GB)** fresh64, cap 16,384, seed 0:
  **42/64 (65.6%)**, GSM8K 84.4% (was 62.5), MATH 46.9% (was 50.0); 1 wrong; 21 cut, **0 end in a verbatim loop**
  (was 9); mean 9,651 tokens. **Paired vs original: +9.4 (SE 4.3, z 2.18), 7 gained / 1 lost → clears 2 SE,
  narrowly.** Graphs `graphs/fresh_m2a_*`. Remaining failures are near-loops cut at 16k.
- **Do-no-harm (passes 6–7, 16 s each, model 1's files):** MMLU 200 (thinking off) **56.5% → 57.5%**, same answer on
  189/200; prose perplexity +0.6%, code +2.1%. No harm seen. Lock/seal decision is for the user.
- **Evidence check (asked by the user):** z 2.18 is thin, the gain is all GSM8K (MATH −3.1), and a failed sealed test
  ends model 2 as "no fix found". Rough pass chance as is ~70–80% (unverified estimate). Chosen: confirm first.
- **Clock:** the 2-day limit (2026-11-18 ~06:00) passed while waiting for the user's reply (no work ran 11-17 ~01:00 →
  11-20). Counted as paused; ~19 h of the 48 used. Flagged to the user.
- **2026-11-20** `problems_confirm64.json` (32 + 32, random.Random(5), excluding all 576 ids used so far). Pass 8
  `confirm_original` then pass 9 `confirm_m2a`, cap 16,384, seed 0. Pooled with fresh64 → 128 paired.
- **2026-11-20 (passes 8–9, ~1 h)** confirm64: original **47/64 (73.4%)** (GSM8K 81.3, MATH 65.6; 16 cut, 4 loops),
  m2a **53/64 (82.8%)** (GSM8K 87.5, MATH 78.1; 11 cut, 0 loops; 0 wrong); mean tokens 8,263 → 6,802.
  **Paired +9.4 (SE 3.7, z 2.55), 6 gained / 0 lost.** MATH improved here (21 → 25), so the fresh64 MATH dip was noise.
  **Pooled 128: 83 → 95, +9.4 (SE 2.8, z 3.33), 13 gained / 1 lost.** Replicated. Ready to lock m2a (user's call).
- **2026-11-20 LOCKED: edit m2a** (user: "yes"). Sealed test started, run once: `data/qwen35-0.8b/holdout/holdout256.json`
  (128 MATH-500 + 128 GSM8K test; burned for model 1 only, never seen by the 2B or used to build m2a), cap 16,384,
  seed 0. Pass 10 original, pass 11 m2a. Pass rule: paired ≥ 2 SE.
- **2026-11-21 SEALED RESULT (passes 10–11, 105 + 96 min):** original **168/256 (65.6%)** (GSM8K 71.9, MATH-500 59.4;
  83 cut, 20 loops; 4 wrong), m2a **188/256 (73.4%)** (GSM8K 84.4, MATH-500 62.5; 59 cut, **0 loops**; 7 wrong);
  mean tokens 9,234 → 7,806. **Paired +7.8 (SE 2.5, z 3.14), 31 gained / 11 lost → PASS.**
- Export: `exports/qwen35-2b/m2a-bf16` (st.patch, 130,294 + 130,268 values; logits identical to the overlay, max
  diff 0.0) → `mlx_lm convert -q --q-bits 4` → `m2a-mlx-4bit` (1.0 GB, 4.503 bpw), answers "42" to 12 + 30.
  First export try failed (copy loop hit `models/Qwen3.5-2B/.cache`, a folder); fixed by copying files only.
  4-bit generation check on fresh64 (pass 12) started.
- **Pass 12 `fresh_4bit` (29 min): 27/64 (42.2%)** (GSM8K 59.4, MATH 25.0); 37 cut (4 loops); mean 13,077 tokens.
  vs original bf16 **−14.1 (SE 6.6) 5/14**; vs m2a bf16 −23.4 (SE 6.2) 2/17. **The 4-bit export loses the gain and
  more.** Unknown: how much is 4-bit itself (model 1 never measured an unedited 4-bit baseline). Pass 13: unedited
  2B at 4-bit (scratchpad conversion), same 64.
- **Pass 13 `fresh_orig_4bit` (unedited 2B, 4-bit, 4.503 bpw): 7/64 (10.9%)**; 57 cut, **56 end in a loop**; mean 14,910
  tokens. vs original bf16 −45.3 (SE 6.3) 0/29. **4-bit quantization alone makes the 2B loop badly.**
  **Edit 4-bit vs unedited 4-bit: +31.2 (SE 6.2, z 5.00), 21 gained / 1 lost.** The edit survives quantization as a
  relative gain; the 4-bit export beats its own unedited 4-bit counterpart. Temporary conversion deleted.
- **MODEL 2 COMPLETE (2026-11-21).** All must-haves met: locked edit m2a, sealed +7.8 (3.14 SE), do-no-harm passed,
  MLX 4-bit export beats unedited 4-bit (+31.2, 5.0 SE). Recommended to run: `exports/qwen35-2b/m2a-bf16`.
  13 of 50 passes, 1 of 3 attempts, ~25 h of 48.
