# Qwen3.5-2B — results

Model 2 of 5, the first blind test: no one has published a fix for this model. Every number below was measured on
this Mac. Anything not yet measured says so.

## Where things stand

| Stage | What | Status |
|---|---|---|
| 1 | Baseline: does the 2B have a defect worth fixing? | **Done — yes, a milder version of model 1's** |
| 2 | Find an edit (at most 3 attempts) | **Done: attempt 1 (edit m2a), locked 2026-11-20** |
| 3 | Sealed test, do-no-harm, export | **Done: sealed test passed (+7.8, 3.14 SE); do-no-harm passed; exported** |

**Model 2 complete (2026-11-21).** Budget used: 13 of 50 passes, 1 of 3 attempts.

## Verdict (sealed test, run once)

256 unseen problems (128 MATH-500 + 128 GSM8K test), cap 16,384, seed 0. Edit m2a was locked before this ran.

| | Original | **Edit m2a** |
|---|---|---|
| Correct | 168 (65.6%) | **188 (73.4%)** |
| GSM8K test / MATH-500 | 71.9% / 59.4% | 84.4% / 62.5% |
| Cut off (exact loops) | 83 (20) | 59 (0) |
| Mean tokens | 9,234 | 7,806 |

**Paired: +7.8 points, standard error 2.5 (3.14 SE), 31 gained, 11 lost. Passes the ≥ 2 SE rule.**

Export: `exports/qwen35-2b/m2a-bf16/` (4.3 GB, identical outputs to the tested edit) and
`exports/qwen35-2b/m2a-mlx-4bit/` (1.0 GB, 4.503 bits per weight; loads and answers correctly).

### Does the gain survive 4-bit? As a relative gain, yes, strongly.

Same 64 practice problems, cap 16,384, seed 0 (passes 1, 5, 12, 13).

| | Original | Edit m2a | Paired |
|---|---|---|---|
| Full size (bf16) | 36 (56.3%) | 42 (65.6%) | +9.4 (SE 4.3) |
| **4-bit** | 7 (10.9%), 56 loops | **27 (42.2%)**, 4 loops | **+31.2 (SE 6.2), 21 gained / 1 lost** |

- 4-bit quantization alone makes the unedited 2B loop on almost everything (−45 points).
- The edit removes most of that: the edited 4-bit model gets about 4 times as many right as the standard 4-bit one.
- **Use the full-size export when memory allows (best: 73.4% sealed). At 4-bit, the edited file is far better than
  the standard 4-bit conversion.**

Limits: 2 days (clock paused 11-17 → 11-20 while waiting for a decision; ~20 h used) · 3 edit attempts · 50 passes.
Budget used: 9 of 50.

## Attempt 1: model 1's recipe (edit m2a)

25 looping answers (passes 1–2) → rewind and replay (pass 3: 94 matched pairs from 25 prefixes) → probe (pass 4) →
top 64 "stuck" writers in layers 15 and 19, reversed at strength 6 → `edits/m2a`.

| Same 64 problems, cap 16,384 | Original | **Edit m2a** |
|---|---|---|
| Correct | 36 (56.3%) | **42 (65.6%)** |
| GSM8K / MATH | 62.5% / 50.0% | 84.4% / 46.9% |
| Cut off (exact loops) | 27 (9) | 21 (0) |
| Mean tokens | 10,459 | 9,651 |

- **Paired: +9.4 points, standard error 4.3 (2.18 SE), 7 gained, 1 lost.** Passes the 2 SE rule, narrowly.
- Exact loops are gone; the 21 still cut are near-loops (second-guessing), as with model 1's edit.
- Graphs: `graphs/fresh_m2a_accuracy.svg`, `_outcomes.svg`, `_lengths.svg`, `_speed.svg`.

### Confirmation on 64 more problems (passes 8–9)

The margin above was thin, and the sealed test runs once, so the same comparison was repeated on 64 new practice
problems (`problems_confirm64.json`) before locking.

| | Original | Edit m2a | Paired |
|---|---|---|---|
| Confirmation 64 | 47 (73.4%) | **53 (82.8%)** | +9.4, SE 3.7, 6 gained / 0 lost |
| **Pooled 128** | 83 (64.8%) | **95 (74.2%)** | **+9.4, SE 2.8 (3.33 SE), 13 gained / 1 lost** |

- Same gain on both halves. MATH rose on the new half (65.6% → 78.1%), so the first half's MATH dip was noise.
- Exact loops at the cap: 4 → 0. Mean answer length 8,263 → 6,802 tokens.

## Do-no-harm (passes 6–7)

| Check | Original | Edit m2a |
|---|---|---|
| MMLU 200, thinking off | 56.5% | 57.5% (same answer on 189/200) |
| Prose / code perplexity | — | +0.6% / +2.1% |

## Baseline (pass 1)

Same 64 practice problems model 1 was tested on, cap 16,384 tokens, same settings and seed.

| | Qwen3.5-0.8B original | **Qwen3.5-2B original** |
|---|---|---|
| Correct | 8/64 (12.5%) | **36/64 (56.3%)** |
| Wrong answer | 1 | 1 |
| Cut off at 16,384 tokens | 55 | 27 |
| … ending in an exact loop | 54 | 9 |

- When the 2B finishes, it is right (36 of 37). **All of its other failures are not finishing.**
- Of the 27 unfinished answers, 9 are exact loops and 18 are near-loops: the same second-guessing model 1's edit A left
  behind ("Wait" 6 times as often as in correct answers; repeated phrases more than twice as often).
- So the target is the same defect as model 1, milder. Upside if it is fixed: up to about 40 points.
