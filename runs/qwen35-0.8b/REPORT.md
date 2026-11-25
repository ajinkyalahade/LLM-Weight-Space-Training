# Qwen3.5-0.8B — results

Last updated 2026-11-14. Every number below was measured on this Mac or read from a named file. Anything not yet
measured says so.

## In one minute

- **The model:** Qwen3.5-0.8B, 0.75 billion weights, 1.6 GB. Small enough to run anywhere.
- **Its defect:** with thinking on, it gets stuck repeating itself and never gives an answer. The blog post reports
  about 89% of math answers cut off this way.
- **The post's fix:** change 130,188 of its weights (0.016%) in two places. Reported result: 7.9% → 48.1% on the
  MATH-500 benchmark (the post), and 10.2% → 59.0% on 256 fresh practice problems (the authors' notes).
- **What I am doing:** (1) reproduce both numbers here, (2) find the same fix with my own tools, (3) try to beat it.

## Verdict — sealed test (run once, 2026-11-16)

**Edit M2 passes.** On 256 problems never used before (128 from MATH-500, 128 from the GSM8K test set, picked at
random after M2 was locked), the original model got 13.3% right and M2 got 52.3%.

![Sealed test](graphs/holdout_accuracy.svg)

| | Original (pass 22) | Edit M2 (pass 23) |
|---|---|---|
| Correct | 34/256 (13.3%) | **134/256 (52.3%)** |
| Correct, GSM8K test / MATH-500 | 17.2% / 9.4% | **70.3% / 34.4%** |
| Wrong answer | 0 | 16 |
| Finished thinking, no answer | 0 | 0 |
| Cut off at 16,384 tokens | 222 | 106 |
| … ending in an exact loop | 199 | **0** |
| Mean tokens per answer | 14,947 | 11,099 |

- **Paired: +39.1 points, standard error 3.2 (12 standard errors; the pass rule is 2). 102 problems gained, 2 lost.**
- Settings: thinking on, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, cap 16,384 tokens, seed 0.
  The problems are listed in `data/qwen35-0.8b/holdout/holdout256.json`. This set is now used and must not be used
  to change the edit.
- The cap is shorter than the post's 32,768, so the absolute numbers are not the post's MATH-500 figure. With more
  room, more of M2's 106 cut-off answers would finish (Finding 9: 46.9% at the full cap on other problems).

## Do-no-harm check — passed

Does the edit hurt anything outside math? Two read-only checks, original vs M2 (passes 24 and 25, `wsp/harm.py`):

| Check | Original | Edit M2 | Change |
|---|---|---|---|
| 200 MMLU knowledge questions, thinking off (19 subjects) | 48.0% | 48.0% | 0.0 points (SE 1.2); 3 gained, 3 lost; same answer on 191 |
| Reading prose (the blog post's text), perplexity | — | — | +0.7% |
| Reading Python code (2 standard-library files), perplexity | — | — | +1.3% |

- No regression beyond noise. The text changes are far smaller than what 4-bit quantization costs.
- Limit: 19 subjects, because the questions came in slices of 10 from one subject each.

## Export

- `exports/qwen35-0.8b/m2-mlx-4bit/` (424 MB): the edited model in MLX 4-bit (4.5 bits per weight), made with
  `mlx_lm.convert -q`. It loads with `mlx_lm.load` and answers ("What is 12 + 30?" → "42").
- Built from a full-precision copy with M2's 128 columns written in; that copy gives exactly the same output as the
  tested edit (logit difference 0.0). The 4-bit build picks the same next token as full precision 82% of the time on
  a test prompt, which is usual for 4-bit.
- Run it: `mlx_lm.generate --model exports/qwen35-0.8b/m2-mlx-4bit --prompt "..."` (thinking is on by default in
  the chat template).

### Does the gain survive 4-bit? Partly.

Same 64 new problems, cap 16,384, seed 0 (pass 26, 18 minutes).

![4-bit check](graphs/quant_accuracy.svg)

| | Original (full precision) | M2 (full precision) | M2 (4-bit export) |
|---|---|---|---|
| Correct | 8/64 (12.5%) | 25/64 (39.1%) | **18/64 (28.1%)** |
| Cut off at 16,384 tokens | 55 | 34 | 29 |
| … ending in an exact loop | 54 | 1 | 20 |

- 4-bit M2 against the original: **+15.6 points (SE 5.6), 12 gained, 2 lost. Passes** the 2-standard-error bar.
- 4-bit M2 against full-precision M2: −10.9 points (SE 6.3). Rounding to 4 bits blurs the edit's small column changes
  and some loops come back (20 of 29 cut-off answers).
- Not measured: an unedited 4-bit original, so how much of the drop comes from 4-bit itself is unknown.
- **Recommendation: for this model, run the full-precision edit** (1.6 GB, rebuilt from the edit file in a minute).
  The 4-bit build saves little memory at this size and keeps only about 60% of the gain.

## Where things stand

| Stage | What | Status |
|---|---|---|
| 1 | Original model fails the way the post says | **Done** — 8.3% correct, 92% cut off (post: 11.5%, 89%) |
| 2 | The authors' edited model fixes it, measured here | **Done** — 49.0% correct; 39 problems gained, 0 lost (Reproduction) |
| 3 | Find the edit myself and compare with theirs | **First result** — same place, same neurons at the top, direction 0.66–0.68 alike (Finding 4) |
| 4 | Try to beat it | **Done: level, not beaten** — over 160 problems my edit M2 and the authors' edit are tied (Findings 8–9) |
| 5 | Sealed holdout, do-no-harm checks, export | **Done** — sealed test 13.3% → 52.3% (+39.1, 12 SE); no harm; 4-bit export keeps part of the gain (+15.6) |

Budget used: 26 of 50 passes (12 generations, 14 probes and checks). **Model 1 meets every must-have of the end goal.**

## Finding 1 — what the authors' edit is, read straight from the two weight files

![The edit reverses each neuron's push along one direction](graphs/published_weights.svg)

- Exactly 2 of the model's 488 tensors differ: the MLP output matrix of **layer 15** and of **layer 19**.
- In each, **64 of 3,584 neurons** are changed. Nothing else in the model is touched.
- Each neuron's output had the same single direction subtracted from it, **6 times over**. So a neuron that pushed
  the model's internal state along that direction by +1 now pushes it by −5. The graph shows all 128 neurons on
  that one line.
- The two layers use nearly the same direction (cosine 0.85).
- The authors' notes call this "top64 alpha6" and say the direction separates *productive reasoning* from
  *repetitive reasoning* (`reference/lm_understand-logs/exp8_math/runs/p50c/work_notes/experiment_history.md`).

| Tensor | Neurons changed | Share of the change in one direction | After ÷ before | Size of change, relative to those columns |
|---|---|---|---|---|
| `layers.15.mlp.down_proj.weight` | 64 | 100.00% | −5.00 | 85% |
| `layers.19.mlp.down_proj.weight` | 64 | 100.00% | −5.00 | 62% |

- The 64 neurons are **not** simply the 64 that point most along the direction (only 31 of 64 in layer 15 and 18 of
  64 in layer 19 are). The authors picked them by how much each one *actually writes* along it while the model is
  looping, which needs the model to be run. That is what stage 3 has to rediscover.

## Finding 2 — my edit tool reproduces their edit closely, not exactly

- Rebuilt with `wsp/edit.py` from the original model: same 2 tensors, same 128 neurons, strength 6.
- 99.84% of the stored values are identical to the authors' file; the rest differ by a rounding step or a few.
- Cause: the authors did not publish their direction file. Mine is estimated from their result.
- Behaviour check on one math prompt (64 positions): my rebuild picks the same next token as the authors' model at
  64 of 64 positions (mean difference in prediction 0.0005; the original model differs from theirs by 0.023).
- Files: `edits/published_rebuild/` (0.5 MB).

## Finding 3 — what this Mac can do with this model

| Answers at once | Tokens per second (short answers) | Memory |
|---|---|---|
| 1 | 137–170 | 1.7 GB |
| 8 | 869 | 2.7 GB |
| 32 | 1,186 | 3.6 GB |
| 64 | 1,439 | 5.9 GB |

- Speed falls and memory grows as answers get long: about 16 KB of memory per token per running answer.
- A looping answer runs to the 32,768-token cap, so one full-length test of 96 problems takes hours, not minutes.
  Measured: the first 16 answers took 23 minutes at 16 at once (390 tokens/s overall).
- **One crash, fixed.** The first full run died after 24 minutes with `Resource limit (499000) exceeded`, 80 s after
  14 new problems were added to a batch whose answers were already 28,000 tokens long. The runner now works in
  separate waves of 16 and saves every answer as it finishes, so a crash loses only the wave in progress.
- Consequence for the plan: exploration will use a shorter cap; the full cap is for final confirmation only.

## Finding 4 — my own tools find the same spot (stage 3, first result)

Method: take 8 of the original model's looping answers, compare the model's internal state while it is still
reasoning with its state once it is looping, and look for the neurons that push it toward the looping state.

![Where the looping state differs from the reasoning state](graphs/mine_a_layers.svg)

- The difference is consistent across the 8 answers in almost every layer (agreement 0.8–0.9) and largest in
  layers 10–14. Layers 15 and 19, where the authors edited, are in the strong region but are not the peak.
- **Direction:** mine and the authors' point the same way, but not exactly (cosine 0.66 in layer 15, 0.68 in layer
  19; two unrelated directions would give about 0.03).
- **Neurons:** my 16 strongest neurons are all among the authors' 64 in layer 15, and 14 of 16 in layer 19. Over
  all 64: 37 shared in layer 15, 30 in layer 19.
- In layer 19 one neuron (number 11) does almost half of the pushing on its own.

## Finding 5 — first attempts to do better (stage 4)

Test: 6 other looping answers, not used to build anything. For each edit, how much less sure the model becomes of
the loop text (wanted) and of the reasoning text before the loop (a cost). This is a quick check that takes
17 seconds; the real test is accuracy on new problems, which comes after stage 2.

![Which part of the edit matters: the direction or the neurons](graphs/swap_confidence.svg)

| Edit | Less sure of loop text | Less sure of reasoning text | Ratio (higher is better) |
|---|---|---|---|
| Authors' edit | 0.085 | 0.057 | **1.50** |
| My edit A (my direction, my 64 neurons, strength 6) | 0.105 | 0.080 | 1.32 |
| Their direction on my neurons | 0.095 | 0.070 | 1.37 |
| My direction on their neurons | 0.063 | 0.051 | 1.24 |
| My edit A on its top 32 neurons only | 0.061 | 0.051 | 1.22 |
| My edit A at strength 3 | 0.020 | 0.030 | 0.66 |

- **Nothing beats the authors' edit yet.** Mine is stronger but less selective.
- **Strength matters a lot:** at strength 3 the edit hurts the reasoning text more than the loop text. The authors
  also only got a large effect at strength 6.
- **Next idea:** the authors built their direction from *matched* pairs — the same text continued once normally and
  once into a loop. Mine compares early text with late text, which also differs in other ways. Matched pairs are the
  next thing to try (Finding 6).

## Finding 6 — matched pairs, tried: worse, because there was too little of it

Method: restart 8 looping answers from 300 tokens before the loop and from the loop's start, 8 times each
(128 continuations of 300 tokens, pass 11, 4 minutes). Keep the starting points where some continuations loop and
others do not, and compare the model's state in the two kinds (pass 12). Build edit M from that (pass 13).

![Matched pairs did not beat the authors' edit](graphs/matched_confidence.svg)

| Edit | Less sure of loop text | Less sure of reasoning text | Ratio (higher is better) |
|---|---|---|---|
| Authors' edit | 0.085 | 0.057 | **1.50** |
| My edit A | 0.105 | 0.080 | 1.32 |
| My edit M (matched pairs) | 0.016 | 0.045 | 0.36 |

- **Only 3 of 16 starting points were usable.** At most points the model's fate is already decided: either all
  8 continuations loop or none do. 34 continuations looped, 39 did not, 55 were in between.
- With 3 points the direction is noisy: the 3 agree only weakly (0.4, against 0.8–0.9 for edit A) and it is less like
  the authors' (cosine 0.54 / 0.52, against 0.66 / 0.68 for edit A).
- The result: edit M barely touches the loop text. **Edit A stays my best candidate.**
- Getting more usable points would cost more runs (more starting points, more samples each). Real accuracy is a
  better use of the budget first: the quick check here is only a stand-in.

## Finding 7 — real accuracy: my edit A stops exact loops but not the endless thinking

Test: 64 new practice problems (`problems_fresh64.json`: 32 GSM8K, 32 MATH, none from the 96), answers capped at
16,384 tokens, same settings and seed for all three models. Passes 14–16, 2026-11-15.

![Accuracy on 64 new problems](graphs/fresh_accuracy.svg)

| | Original (pass 16) | Authors' edit (pass 15) | My edit A (pass 14) |
|---|---|---|---|
| Correct | 8/64 (12.5%) | **21/64 (32.8%)** | 11/64 (17.2%) |
| Correct, GSM8K / MATH | 18.8% / 6.2% | 46.9% / 18.8% | 25.0% / 9.4% |
| Cut off at 16,384 tokens | 55 | 37 | 53 |
| … of those, ending in an exact loop | 54 | 0 | **1** |
| … of those, that closed the thinking section | — | 4 | 1 |
| Mean tokens per answer | 14,827 | 13,296 | 15,466 |

- **Edit A loses clearly.** Against the original: +4.7 points (standard error 4.1), which is within noise
  (5 problems gained, 2 lost). The authors' edit: +20.3 points on the same problems.
- **Why:** edit A removes exact repetition (54 → 1 looping answers) but the model still does not stop. It reaches
  the right answer and then second-guesses it forever, each line slightly different: "Wait. / Wait. / If A=36,
  B=38 … So the age of the older friend is 38. / Wait." My direction learned *"this text repeats"*; the authors'
  learned something closer to *"this reasoning has stopped going anywhere"*.
- The quick check (Finding 5) ranked the edits correctly but hid how large the gap is.
- The 16,384 cap is short for this model: the authors' edit gets 32.8% here but 49.0% at 32,768 (on other problems).
  All three models had the same cap, so the comparison is fair.
- Edit A's pass took 137 minutes, against 22–24 for the others. Its log shows 8 pauses of 5–17 minutes with no
  output at all: the process was suspended (unverified: the Mac likely slept despite `caffeinate -i`). Answers are
  not affected.

![What happened to each answer](graphs/fresh_outcomes.svg)

## Finding 8 — my edit M2 matches the authors' edit

Method: the authors' approach, done with enough data. 40 looping answers of the original (none used before), each
restarted 900, 600 and 300 tokens before its loop, 8 times (960 restarts of 300 tokens, pass 17). 18 starting
points produced both a looping and a non-looping restart. The direction is the difference between the two kinds
(pass 18); the edit is the same shape as theirs: 64 neurons in layers 15 and 19, strength 6 (`edits/mine_m2/`).

- Direction: similarity with the authors' 0.76 (layer 15) and 0.75 (layer 19); edit A had 0.66 / 0.68.
- Neurons: all 16 of my top 16 are among theirs in layer 15; 13 of 16 in layer 19.
- The quick check scored it 0.50, the lowest of all. That check rewards breaking exact repetition, which is not
  the same thing as getting answers right (Finding 7). It is no longer used to rank edits.

Same 64 new problems as Finding 7, same cap, settings and seed (pass 20, 20 minutes):

![Accuracy on 64 new problems, four models](graphs/fresh_accuracy.svg)

| | Original | Authors' edit | My edit A | **My edit M2** |
|---|---|---|---|---|
| Correct | 8/64 (12.5%) | 21/64 (32.8%) | 11/64 (17.2%) | **25/64 (39.1%)** |
| Correct, GSM8K / MATH | 18.8% / 6.2% | 46.9% / 18.8% | 25.0% / 9.4% | 53.1% / 25.0% |
| Cut off at 16,384 tokens | 55 | 37 | 53 | 34 |
| … ending in an exact loop | 54 | 0 | 1 | 1 |
| Mean tokens per answer | 14,827 | 13,296 | 15,466 | 11,873 |

Paired on the same problems:

| M2 against | Difference | Standard error | Problems gained / lost | Verdict |
|---|---|---|---|---|
| Original | +26.6 points | 6.0 | 18 / 1 | Better (4.4 SE) |
| My edit A | +21.9 points | 5.2 | 14 / 0 | Better (4.2 SE) |
| Authors' edit | +6.2 points | 5.4 | 8 / 4 | **Not established** (1.2 SE; the rule is 2) |

- **M2 is as good as the authors' edit, possibly a little better.** 64 problems cannot tell a 6-point gap from luck.
- It answers faster: 11,873 tokens on average, against 13,296.
- Lesson: the edit's shape (layers, neuron count, strength) was right from the start; what mattered was what the
  direction was learned from. "Restarts of the same text that loop vs ones that don't" beats "early text vs late
  text", and it needs enough restarts (3 usable points failed, 18 worked).

## Finding 9 — at full length, M2 and the authors' edit are tied

Edit M2 on the original 96 problems at the full 32,768-token cap, same settings and seed as stage 2 (pass 21,
66 minutes, 2026-11-15).

![Accuracy on the original 96 problems](graphs/m2_96_accuracy.svg)

| | Original (pass 1) | Authors' edit (pass 10) | My edit M2 (pass 21) |
|---|---|---|---|
| Correct | 8/96 (8.3%) | **47/96 (49.0%)** | **45/96 (46.9%)** |
| Correct, GSM8K / MATH | 9.4% / 7.8% | 50.0% / 48.4% | 43.8% / 48.4% |
| Wrong answer | 0 | 16 | 20 |
| Finished thinking, no answer | 0 | 14 | **0** |
| Cut off at 32,768 tokens | 88 | 19 | 31 |
| … ending in an exact loop | 87 | **0** | 5 |
| Mean tokens per answer | 30,411 | 18,919 | 20,242 |

- M2 against the original: +38.5 points (standard error 5.0), 37 gained, 0 lost.
- M2 against the authors' edit: −2.1 points (standard error 4.2), 7 gained, 9 lost: **within noise**.
- **Both tests together, 160 problems:** M2 70 correct, the authors' edit 68; 15 gained, 13 lost. **Tied.**
- Each edit fails differently. M2 never stops thinking without answering, but on very long answers 5 still slip back
  into a loop. The authors' edit never loops, but 14 times it ends its thinking and writes no answer.
- Verdict for stage 4: **an independent edit, found from scratch with my own tools, matches the published one.** It
  does not beat it.

## Reproduction — the authors' edit works on this Mac (stages 1 and 2)

Same 96 practice problems as the authors' first run (`runs/p50c/requests/fwd_1.json.claimed`), same settings
(thinking on, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5), at most 32,768 new tokens per answer,
seed 0. Passes 1 and 10 in `passes.jsonl`. Measured 2026-11-15.

![Accuracy: original 8%, authors' edit 49%](graphs/reproduction_accuracy.svg)

| | Original (pass 1) | Authors' edit (pass 10) | Post's run of the original |
|---|---|---|---|
| Correct | **8/96 (8.3%)** | **47/96 (49.0%)** | 11/96 (11.5%) |
| Correct, GSM8K | 9.4% | 50.0% | 6.3% |
| Correct, MATH | 7.8% | 48.4% | 14.1% |
| Wrong answer | 0 | 16 | — |
| Finished thinking, gave no answer | 0 | 14 | — |
| Cut off at the token cap | 88 | 19 | 85 |
| Cut-off answers that end in a verbatim loop | 87 of 88 | **0 of 19** | — |
| Mean tokens per answer | 30,411 | 18,919 | 29,427 |
| Time for all 96 | 87 min | 58 min | — |

- **Paired on the same problems: +40.6 points (standard error 5.0). 39 problems gained, 0 lost.**
- The original reproduces the post: 8 vs 11 correct is within chance for 96 problems (about ±3).
- The edit matches what the authors report: 7.9% → 48.1% on MATH-500 (post), 10.2% → 59.0% on 256 practice
  problems (their notes).
- **The edit removes loops completely in this test.** 19 answers still hit the cap, but none of them ends in a
  repeated block of text. They are long, real attempts.
- **What is left to fix:** 16 wrong answers and 14 that finished thinking without writing an answer. These are the
  targets for stage 4. The second kind is a new failure that the original never reached.

![What happened to each answer](graphs/reproduction_outcomes.svg)

![How long the answers are](graphs/reproduction_lengths.svg)

- The original's answers are either short and correct or exactly 32,768 tokens (looping). The edited model's lengths
  are spread out: it reasons as long as it needs to.

![Generation speed](graphs/reproduction_speed.svg)

- Speed falls from about 850 to about 320 tokens/s within each wave of 16 answers as they get longer, then jumps back
  up when a new wave starts. The original's line is the last of its three sessions (it was resumed after a crash).
  Its slow stretch at minutes 2–20 is when a second GPU job ran beside it.

## What is in this folder

| Path | What |
|---|---|
| `REPORT.md` | This page |
| `notes.md` | Lab notebook: exact settings, every decision, dated |
| `graphs/` | The graphs above. Remake with `wsp/report.py` |
| `passes.jsonl` | One line per test run (the budget ledger) |
| `passes/gen_<k>_<label>/` | Every answer of run *k*, compressed, plus its summary and speed log |
| `edits/<id>/` | One edit: what was asked (`spec.json`), what changed (`edit.json`), the changed columns |
| `edits/mine_a/localise.json` | How edit A was found: per-layer numbers and the ranked neurons |
| `published_u15.npy`, `published_u19.npy` | The direction of the authors' edit in each layer, extracted from their file |
