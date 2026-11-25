# Model 1 — Qwen3.5-0.8B — lab notebook

Read this first every session. Newest entries at the bottom. Every number here was measured or read from a file on the
date given; nothing is recalled.

## Fixed facts

- **Base:** `Qwen/Qwen3.5-0.8B` at commit `2fc06364715b967f1860aea9cf38778875588b17` → `models/Qwen3.5-0.8B/`.
  Same commit the post's edit was made on (`reference/.../exp8_math/README.md`).
- **Answer key:** `RikiyaT/Qwen3.5-0.8B-weight-space-programming` → `models/Qwen3.5-0.8B-wsp-published/`.
- **Size:** 488 tensors, 873.4M values, all bfloat16. Language model 752.4M (embedding 254.3M of that, tied to the
  output layer — there is no separate `lm_head` tensor). Vision and `mtp` tensors are ignored by mlx-lm.
- **Shape:** 24 layers, width 1,024, MLP width 3,584. Layers 3, 7, 11, 15, 19, 23 are full attention
  (`self_attn.{q,k,v,o}_proj`); the other 18 are linear attention (`linear_attn.*`). Every layer has
  `mlp.{gate_proj, up_proj, down_proj}`. `down_proj` is (1024, 3584): **one column per MLP neuron**.
- **Not editable under our rules:** `embed_tokens.weight`, `norm.weight`.

## Protocol `think35` (must never change between base and edited)

- Chat template with `enable_thinking=True`; the prompt ends in `<think>\n`.
- User message = problem + `"\nPlease reason step by step, and put your final answer within \boxed{}."`
- temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, at most 32,768 new tokens.
- Grading: last `\boxed{}` after the last `</think>`, checked with math-verify 0.9.0. A cut-off answer is a failure.

### Presence penalty — a difference from mlx-lm's default

mlx-lm's built-in presence penalty covers only the last 20 tokens and includes the prompt. `wsp/lm.py` uses its own:
every generated token, no prompt tokens. **Unverified** that this matches vLLM exactly; the check is whether pass 1
reproduces the post's base numbers.

## The published edit, read from the two files (2026-11-14, `wsp/diff.py`)

- Exactly 2 tensors differ: `layers.15.mlp.down_proj.weight` and `layers.19.mlp.down_proj.weight`.
- 64 columns changed in each; 65,260 + 64,928 = 130,188 values (the README's number).
- Each change is rank one: a single direction subtracted from those 64 columns (first singular value holds 100.00%).
- Size: 85% of the norm of the changed columns in layer 15, 62% in layer 19 (10.6% and 8.1% of the whole tensor).
- Layer 15 columns: 0, 2, 11, 25, 34, 35, 38, 39, 46, 50, 53, 56, 61, 62, 63, 64, 67, 73, 100, 108, 129, 135, 156,
  169, 209, 215, 264, 266, 294, 327, 328, 357, 371, 376, 412, 414, 442, 504, 505, 509, 516, 580, 590, 601, 638, 682,
  738, 882, 926, 930, 1047, 1060, 1087, 1130, 1155, 1184, 1460, 1481, 1787, 1943, 2005, 2066, 2738, 3003
- Layer 19 columns: 1, 5, 8, 11, 13, 15, 17, 18, 34, 36, 44, 55, 63, 66, 69, 92, 93, 95, 98, 114, 119, 127, 145, 152,
  161, 168, 171, 195, 246, 259, 265, 343, 364, 445, 506, 544, 621, 625, 636, 744, 766, 770, 799, 877, 965, 1003, 1346,
  1410, 1505, 1704, 1799, 2072, 2135, 2175, 2248, 2314, 2376, 2437, 2514, 2595, 2664, 2684, 2816, 3343
- **I have seen this answer.** Stage 3 (re-deriving it) is therefore informed, not blind.

## Speed on this Mac (2026-11-14, `wsp/speed.py`, 400 new tokens per answer)

| Batch | Tokens/s total | Peak memory |
|---|---|---|
| 1 | 137–170 | 1.65 GB |
| 8 | 869 | 2.7 GB |
| 32 | 1,186 | 3.6 GB |
| 64 | 1,439 | 5.9 GB |

Speed at long context: not measured yet — read it from `passes/gen_1_original/progress.log`.

## Targets to reproduce (from the post's own first pass, `runs/p50c/requests/fwd_1.done.json`)

Same 96 problems (32 GSM8K, 64 MATH), 1 sample, 32,768-token cap, base model:
accuracy 11.46% (GSM8K 6.25%, MATH 14.06%), cut off 88.54%, mean 29,427 tokens, wrong answers 0.

## Log

- **2026-11-14** Phase 0: environment, downloads, `wsp/{st,diff,lm,speed,grade,generate}.py`. Smoke test and a
  4-problem trial passed (`passes/gen_1_original_trial/`, not counted).
- **2026-11-14 20:5x** Launched pass 1 (base) and pass 2 (published edit) on the 96 problems at batch 32.
  **Stopped after 13 minutes, no results.** Two faults in my runner: answers were only saved at the end, and memory
  at batch 32 was heading for 20–25 GB (5.4 GB at 5.7k tokens, 7.4 GB at 9.5k, a jump to 12.2 GB when one answer
  finished and the batch was rebuilt).
- **2026-11-14 21:2x** `wsp/generate.py` now writes each answer as it finishes, resumes when the same command is run
  again (tested: interrupted after 3 of 6, finished 6), compresses the answers at the end, and takes `--edit`.
  Relaunched passes 1–2 at **batch 16**. Output: `passes/pass1.out`, `passes/pass2.out`. Results: _pending_.
  Measured at batch 16: 373 tokens/s with answers 12.7k tokens long, peak 5.5 GB (while another small job ran).
- **2026-11-14** The published edit, analysed (`wsp/report.py weights`, graph `graphs/published_weights.svg`):
  it is `W[:, cols] -= 6 · u (uᵀ W[:, cols])` on 64 columns per layer. Per-column strength 6.0006 (L15) and 6.0004
  (L19), range 5.991–6.007. Every column's projection on `u` is multiplied by −5: mean size 0.0306 → 0.1531 (L15),
  0.0240 → 0.1201 (L19). Cosine between the two layers' directions: 0.849. The 64 columns are not the top 64 by
  |uᵀW| (31/64 in L15, 18/64 in L19; weakest chosen ranks 1,114 and 2,420 of 3,584). The authors' notes
  (`work_notes/experiment_history.md`) call it "top64 alpha6": `u` is "a unit repetition direction from FOUR matched
  productive/repetitive original contexts", columns from `repetition_writer_top64.pt`. Neither file is published.
  Their dose history: full-direction alpha 1 → 2 → 3 → 4, then top-64 alpha 4, then top-64 alpha 6 (kept).
- **2026-11-14** `wsp/edit.py` built and tested by rebuilding the published edit (`edits/published_rebuild/`):
  3,664,091 / 3,670,016 (L15) and 3,663,854 / 3,670,016 (L19) stored values identical; 4,961 + 5,117 one step off;
  964 + 1,045 more than one step off. Not bit-exact because `u` is estimated from the rounded result.
  Behaviour on one prompt (64 positions): rebuild vs published same top token 64/64, mean KL 4.9e-4;
  base vs published 61/64, mean KL 2.3e-2. Refusals tested: embedding edit refused; whole-tensor ×1.5 refused
  (first version missed it — tolerance 0.1% was tighter than bfloat16's 0.4%; now 1%).
- **2026-11-14** User rules: one model at a time, no clutter, results folder with notes and graphs → `REPORT.md`,
  `graphs/`, `wsp/chart.py`, `wsp/report.py`. Edits are stored as changed columns only.

- **2026-11-14 21:44** Relaunched pass 1 (batch 16, continuous refill) **crashed at 1,430 s** with
  `RuntimeError: [metal::malloc] Resource limit (499000) exceeded` (from `mx.async_eval` in mlx-lm
  `GenerationBatch._step`). 17 answers were saved (2 correct, 14 cut at 32,768 with a verbatim loop, 1 correct at
  2,454). Timeline: 14 answers hit the cap at ~1,350 s, 14 new prompts were merged into the running batch (mlx-lm
  `BatchKVCache.extend` left-pads newcomers to the longest running answer, ~28k), speed fell to 148 tokens/s, crash
  80 s later. MLX `allocator.cpp` (v0.32.4) frees the whole buffer cache before raising, so ~499k buffers were live.
  Measured: 16 answers at 1,500 tokens each hold 454 live buffers with my presence penalty, 438 without — no
  per-token leak. Root cause not found. Fix: waves of 16 with a fresh `BatchGenerator` each (no refill), and a
  retry loop (4 attempts) that resumes from saved answers. Relaunched 2026-11-15 as one background job: pass 1
  continues, then the published model (`passes/original.out`, `passes/published.out`).
- **2026-11-15** Loop onsets in the 14 saved cut-off answers (`wsp/loops.py`, 8-gram repeat ≥ 90% over 200 tokens):
  median 5,561 tokens, earliest 3,691, latest 10,092; after onset 100% of tokens repeat. Correct answers: 2,454,
  7,325, 7,490 tokens. ⇒ A cap of ~12k–16k sees nearly every loop start and every correct answer so far.
- **2026-11-15 pass 2 (`fwd_2_loops_a`)**: probe on the 8 earliest-onset answers (46,171 tokens, 14 s, 2.9 GB).
  Groups: productive = answer tokens 20 … onset−150; repetitive = onset+100 … onset+1,100. Loop text log-prob −0.003
  to −0.029 per token vs −0.48 to −0.64 for reasoning. `wsp/localise.py` → `edits/mine_a` (direction = mean
  difference, top 64 writers by mean activation × uᵀ·column, strength 6). Cosine with the published u: 0.661 (L15),
  0.677 (L19). Shared neurons: 37/64 (L15, all 16 of my top 16), 30/64 (L19, 14 of 16). Agreement between cases
  (mean pairwise cosine of d) 0.73–0.89 in layers 1–22; |d|/|resid| peaks at layers 10–14 (0.80–0.87).
  L19 neuron 11 writes +0.311 of +0.717 for the top 64.
- **2026-11-15 passes 3–9**: same request (`fwd_3_b_original/request.json.gz`: the next 6 looping answers, none
  used for the direction) on original, published and 5 edits. Mean change in log-prob per token
  (loop text / reasoning text, ratio): published −0.085 / −0.057 (1.50); mine_a −0.105 / −0.080 (1.32);
  mine_s3 −0.020 / −0.030 (0.66); mine_top32 −0.061 / −0.051 (1.22); swap_dir (their u, my neurons)
  −0.095 / −0.070 (1.37); swap_cols (my u, their neurons) −0.063 / −0.051 (1.24).
  Reading (mine, not verified by generation): strength ≥ 6 is needed; their u is better than mine, and their neurons
  only help together with their u. Their u came from matched productive/repetitive continuations of the same
  prefix; mine compares early vs late text of one answer, which also differs in position and topic.
  Caveat: teacher-forced log-prob on 6 answers is a proxy; only generation decides.

- **2026-11-15 09:5x** First matched-pairs run (`wsp/branch.py`, 32 prefixes of 4–10k tokens, 8 prefilled at once)
  next to the running full-length pass: **GPU out of memory** after 32 of 128 continuations
  (`kIOGPUCommandBufferCallbackErrorOutOfMemory`). Not counted; its empty folder removed. The main pass survived but
  slowed from ~400 to ~150 tokens/s while both ran. Rule from now on: **one GPU job at a time** (probes of a few
  seconds excepted). `branch.py` now uses 16 at once, prefill 2, and makes its folder only on success.
  Labelling check on real text (gsm8k_train_2364): 8-gram repeat share 0.93 right after the loop onset, 0.46 in
  normal reasoning 600–300 tokens before it ⇒ thresholds: loops ≥ 0.85, does not ≤ 0.55.
  `wsp/localise.py --matched` added; tested on synthetic data (recovers a planted direction, cosine 1.000) and the
  old mode gives unchanged numbers.

- **2026-11-15 ~10:55** Pass 1 (original) finished on its resumed session: 87 min for the last 79 answers, no crash
  with waves of 16. **8/96 correct (8.3%)**; GSM8K 9.4%, MATH 7.8%; 88 cut at 32,768, 87 of them end in a verbatim
  loop; mean 30,411 tokens; peak 12.3 GB. Post's run: 11/96, 85 cut, mean 29,427: within chance.
- **2026-11-15 ~12:00** Pass 10 (authors' edit, `models/Qwen3.5-0.8B-wsp-published`), same 96, same seed, 6 waves,
  58 min, attempt 1. **47/96 correct (49.0%)**; GSM8K 50.0%, MATH 48.4%; 16 wrong, 14 finished thinking with no
  answer, 19 cut at the cap, **0 of the 19 end in a loop**; mean 18,919 tokens; peak 8.6 GB.
  Paired vs pass 1: +40.6 points (SE 5.0), 39 gained, 0 lost. **Reproduction confirmed.**
  Remaining failures (16 wrong, 14 no answer, 19 long-but-not-looping) are the stage 4 targets.
  `wsp/report.py` `progress()` now keeps only the last session of a resumed run (the clock restarts at 0 per session;
  the speed graph drew lines back in time).
- **2026-11-15 ~12:25** Pass 11 `gen_11_branch` (matched pairs, 228 s, peak 4.9 GB, GPU alone): 128 continuations,
  34 rep / 39 prod / 55 unclear; only **3 prefixes with both** (gsm8k_train_3513 @−300 and @0, gsm8k_train_1928 @0)
  ⇒ 11 cases. Most prefixes are decided (all loop or none).
  Pass 12 `fwd_12_matched_a`: **1,215 s** for ~66k tokens (a similar probe took 14 s). Unverified cause: run without
  `caffeinate`, Mac likely slept; pass 13 with caffeinate took 18.5 s. Rule: every GPU job under `caffeinate -i`.
  `localise --matched`: agreement 0.27–0.47 per layer; cosine with published u 0.543 (L15), 0.516 (L19); neurons in
  common 33/64 and 20/64. Edit `mine_m` (strength 6, top 64). Pass 13 probe on held-back loops: loop −0.016,
  reasoning −0.045, ratio 0.36: **worse than every earlier variant**. Not pursued further for now.
  All edits' neurons cluster at low indices (55–83% below 512, the published one too): a property of the model.
- **2026-11-15 ~12:40–16:00** Fresh test, `problems_fresh64.json` (random.Random(1), 32 gsm8k_train + 32 math_train,
  none of the 96), cap 16,384, batch 16, seed 0. Pass 14 mine_a 11/64 (17.2%), cut 53 (1 loop, 1 closed thinking);
  pass 15 published 21/64 (32.8%), cut 37 (0 loops, 4 closed thinking); pass 16 original 8/64 (12.5%), cut 55
  (54 loops). mine_a vs original +4.7 (SE 4.1), 5 gained / 2 lost: **not established**.
  Reading: mine_a turns verbatim loops into near-loops ("Wait. / Wait." second-guessing after the right answer);
  4-word repeat share of the last 6,000 chars of cut answers 0.37 (mine_a) vs 0.27 (published). The early-vs-late
  contrast captures surface repetition, not "stuck". Pass 14 took 137 min: 8 log gaps of 5–17 min (process
  suspended; Mac sleep likely, under caffeinate -i; unverified).
- **2026-11-15 evening** Matched pairs, bigger: `loops.py --skip 14 --cases 40` on pass 1 (the 40 looping answers
  after the 14 used for edit A and the held-back set; 289,334 tokens), `branch.py --offsets -900 -600 -300`, n 8,
  length 300. Pass 17 `gen_17_branch_b`: 960 continuations in 1,302 s; 149 rep / 463 prod / rest unclear;
  **18 prefixes with both** ⇒ 80 cases. Pass 18 `fwd_18_matched_b` probe: 5,008 s (should be ~3 min; pauses again).
  `localise --matched` → `edits/mine_m2`: agreement 0.19–0.33 per layer (prefix-level pairs are noisier than
  early-vs-late); **cosine with published u 0.757 (L15), 0.749 (L19)**, the closest yet (A: 0.661 / 0.677);
  neurons in common 36/64 (all 16 of top 16) and 22/64 (13 of 16). L19 neuron 11 again the largest writer.
  Pass 19 quick check on held-back loops: loop −0.030, reasoning −0.059, ratio 0.50 (A 1.32, published 1.50).
  The quick check rewarded A, which then failed in generation (Finding 7), so it is not trusted to rank these;
  M2 goes to the fresh-64 generation test (pass 20).
  Shell lesson, again: in zsh `$P` holding "python -I" is not split; first chain launch ran nothing. Use bash -c.
- **2026-11-15 ~17:50–18:10** Pass 20 `gen_20_fresh_mine_m2` (fresh64, cap 16,384, seed 0, 20 min once awake):
  **25/64 (39.1%)**, GSM8K 53.1%, MATH 25.0%; cut 34 (1 loop); mean 11,873 tokens; peak 6.3 GB.
  Paired: vs original +26.6 (SE 6.0) 18/1; vs mine_a +21.9 (SE 5.2) 14/0; **vs published +6.2 (SE 5.4) 8/4 →
  not established**. Sleep verified: `pmset -g log` shows "Wake from Deep Idle … UserActivity" at 17:50:37, with
  caffeinate -i asserted, so caffeinate -i alone does not keep this Mac running (lid/user-initiated sleep).
  `wsp/report.py passes --names` added (legend overflowed with 4 long labels).
- **2026-11-15 ~18:25–19:30** Pass 21 `gen_21_m2_96` (the 96, cap 32,768, seed 0, `caffeinate -dims`, 66 min, no
  pauses): **45/96 (46.9%)**, GSM8K 43.8%, MATH 48.4%; 20 wrong, 0 no-answer, 31 cut (5 end in a loop); mean 20,242
  tokens; peak 9.7 GB. vs original +38.5 (SE 5.0) 37/0; **vs published −2.1 (SE 4.2) 7/9**.
  Pooled with fresh64: M2 70/160, published 68/160, 15 gained / 13 lost ⇒ tied. Stage 4 closed: matched, not beaten.
  M2's residual loops appear only past 16k tokens (1 of 34 cut at 16k, 5 of 31 at 32k).
  End-goal proposal (limits per model; model 1 to lock, sealed test once, export) sent to the user, awaiting a yes.
- **2026-11-16 ~02:15** User: "yes". End goal + limits written at the top of `plan.md` and in memory. **M2 locked.**
  Holdout: MATH-500 (HuggingFaceH4/MATH-500 test.jsonl, 500) + GSM8K test (openai/grade-school-math test.jsonl,
  1,319), converted to `data/qwen35-0.8b/holdout/{math500,gsm8k_test}.jsonl` (raw downloads deleted);
  `holdout256.json` = random.Random(2026) 128 + 128. Cap 16,384 (full set at 32k would be ~38 h).
- **2026-11-16 02:20–05:05** Sealed test, run once. Pass 22 original: 34/256 (13.3%), 222 cut (199 loops), 96 min.
  Pass 23 M2: **134/256 (52.3%)**, GSM8K 70.3%, MATH-500 34.4%, 16 wrong, 106 cut (0 loops), mean 11,099 tokens,
  75 min. **+39.1 (SE 3.2), 102 gained / 2 lost: PASSED.** Holdout now burned.
  Next: do-no-harm (non-math knowledge + text likelihood, base vs M2), 4-bit export + check, close model 1.
- **2026-11-16 ~05:15** Do-no-harm: `wsp/harm.py` (new). Data `data/qwen35-0.8b/harm/`: mmlu200.jsonl (cais/mmlu
  via datasets-server rows API, 20 slices of 10 at random.Random(7) offsets ⇒ 19 subjects; raw deleted), prose.txt
  (24k chars of the post's text), code.txt (json/decoder.py + textwrap.py, 24k chars). Pass 24 original, pass 25 M2:
  MMLU 48.0% → 48.0% (paired 0.0, SE 1.2, 3/3, same letter 191/200); prose logp −3.2220 → −3.2293 (ppl +0.7%);
  code −1.6266 → −1.6395 (ppl +1.3%). **Passed.** (zsh `${ed:+--edit $ed}` not split: the M2 run failed once with
  "unrecognized arguments" before doing anything.)
- **2026-11-16 ~05:25** Export: `exports/qwen35-0.8b/m2-bf16` (st.patch; 65,091 / 65,120 values changed; logits
  identical to the overlay, max diff 0.0) → `mlx_lm.convert -q --q-bits 4` → `m2-mlx-4bit` (424 MB, 4.508 bpw);
  top-token agreement with full precision 0.824 on a test prompt; answers "42" to 12 + 30. 4-bit generation check on
  fresh64 started (pass 26). m2-bf16 (1.6 GB) is a temporary step, to delete once the user agrees.
- **2026-11-16 ~05:50** Pass 26 `gen_26_fresh_m2_4bit`: **18/64 (28.1%)**, 16 wrong, 29 cut (20 loops), 18 min.
  vs original +15.6 (SE 5.6) 12/2 ⇒ passes the ≥2 SE bar; vs M2 bf16 −10.9 (SE 6.3) 5/12. 4-bit keeps ~60% of the
  gain; loops partly return. No unedited 4-bit baseline measured. Recommendation in REPORT: use full precision for
  this model. **Model 1 meets all must-haves.** Remaining: user's OK on deleting m2-bf16 + the published model.

## Next (in order)

1. ~~Stage 1–2 results~~ done (REPORT.md, Reproduction).
2. ~~Matched-pair direction~~ done, failed (3 prefixes too few). Old text kept: Matched-pair direction, once the GPU is free (commands ready, ~3 min of GPU):
   `wsp/branch.py --source passes/fwd_2_loops_a/request.json.gz --out <req>` → `wsp/probe.py --request <req>` →
   `wsp/localise.py --matched --out edits/mine_m` → `wsp/edit.py` → probe on `fwd_3_b_original/request.json.gz`.
3. Generation test at a 12k–16k cap on fresh practice problems (not the 96): original vs published vs best of mine.

## Decisions

- **Exploration cap.** A full-length pass (32,768 tokens) costs hours here. The authors themselves explored at
  16,384 (gen41/gen42) and saw the same base failure (89–90% cut off). Stage 3–4 exploration will use a shorter
  cap, chosen after pass 1 shows how long correct answers are; the full cap is kept for confirmation and holdout.
- **Batch 16** for full-length passes on this Mac (memory), 32–64 for short ones.

## Tools (all built 2026-11-14)

| Tool | Does | Tested |
|---|---|---|
| `wsp/st.py` | reads/writes weight files, exact bfloat16 | yes |
| `wsp/diff.py` | which weights differ between two models | yes |
| `wsp/edit.py` | applies an edit spec; saves changed columns only; refuses out-of-scope edits | yes — rebuild + 2 refusals |
| `wsp/lm.py` | load (with an edit on top), protocol, presence penalty, pass numbering | yes |
| `wsp/generate.py` | generation pass, saves as it goes, resumes | yes — interrupt + resume |
| `wsp/grade.py` | answer extraction and grading | yes (trial) |
| `wsp/probe.py` | teacher-forced pass: residual stream, MLP activations, token log-probs; saves group averages | yes — matches mlx-lm exactly (difference 0.00); 5,121 tokens in 1.9 s |
| `wsp/loops.py` | finds where an answer starts looping; builds a probe request | runs; onset rule not yet checked on full-length answers |
| `wsp/chart.py`, `wsp/report.py` | graphs (SVG) and tables for `REPORT.md` | yes — rendered and inspected |

Probe conventions: `logp[t]` is for token `t` given the tokens before it; the state that produced token `t` is at
position `t − 1`; `resid` row 0 is the embedding, row `L + 1` is after layer `L`; `act` is the input of `down_proj`.
Attention patterns are not captured yet (add when a head-level question comes up).

Observation (2026-11-14, 16 test answers cut at 2,500 tokens, not a counted pass): only 1 of 16 is in a strict loop
by 2,500 tokens (onset 2,097). Loops mostly start later than that.
