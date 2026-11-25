# Model 3 — MiniCPM5-2B (coding, LiveCodeBench) — report

Started 2026-11-21. Limit 2026-11-23. Details and every decision: `notes.md`. Passes: `passes.jsonl`.

| Stage | Status |
|---|---|
| Data, splits, sandbox | done (217 AtCoder stdin problems; practice 64 / collect 25 / sealed 128; sandbox selftest 8/8) |
| Baseline (pass 1) | done: 21/64 |
| Diagnosis | done: never stops thinking; forced close rescues 10 of 42 |
| Edit attempts (≤ 3) | 3 used: m3a, m3b failed; **m3c passes on practice64** |
| Confirmation | done: seed-1 replicate 23 → 29 (z 2.55); pooled 128 answers 44 → 59 (z 3.83) |
| Sealed test (once) | **PASS: 43 → 67 of 128, +18.8 pts, SE 3.5, z 5.41, 24 gained / 0 lost** (m3c locked 2026-11-22, sha256 `ae12512e…`) |
| Do-no-harm | **passed**: MMLU 52.5% → 52.5% (200/200 same answers); perplexity prose +0.00%, code +0.03% |
| Export | bf16 `exports/minicpm5-2b/m3c-bf16` (logits = overlay); MLX 4-bit `m3c-mlx-4bit` beats unedited 4-bit on practice64: 17 → 25, +12.5 pts, z 3.00, 8 / 0 |
| **Verdict** | **MODEL 3 COMPLETE (2026-11-22). Fix found: m3c.** 18 of 50 passes, 3 of 3 attempts, ~27 h of 48 |

**Recommended to run:** `exports/minicpm5-2b/m3c-bf16`. The 4-bit export keeps the relative gain but writes a stray
`</think>` in 11 of 64 answers (0 at bf16). Graphs: `graphs/sealed_m3c_*`, `graphs/practice_4bit_*`.

## Baseline (pass 1, practice 64, thinking on, temp 1.0, top-p 0.95, cap 16,384)

| | easy (15) | medium (15) | hard (34) | all (64) |
|---|---|---|---|---|
| correct | 15 | 6 | 0 | **21 (32.8%)** |
| cut at cap | 0 | 8 | 34 | 42 |
| wrong answer | 0 | 1 | 0 | 1 |

All 42 cut answers are still thinking at the cap, and none ends in a loop (repeated-8-gram share in the last 2,000
tokens < 0.2 in 41 of 42). The model does not get stuck repeating; it keeps re-checking and never commits.

## Forced close (pass 2): the ceiling of a "stop and answer" edit

Each cut answer's thinking stopped at K tokens, `</think>` appended, the model writes the answer (≤ 4,096 tokens).

| K | correct of 42 cut | medium | hard |
|---|---|---|---|
| 8,192 | 6 | 4 | 2 |
| 16,384 | 10 | 7 | 3 |

## Edit attempts (one row: the `</think>` row of `lm_head`, 2,048 values; fitted on the collect set only)

| | what was added to the row | practice64 | vs original 21/64 | why it failed |
|---|---|---|---|---|
| m3a | 0.12 · "done" direction (mean difference, line ends only) | stopped after 13 answers | — | `</think>` mid-sentence from the first tokens: mid-line positions were never in the calibration |
| m3b | 0.3 · whitened "done" + 1.0 · whitened "late line end" | 12/64 | −14.1 pts, z −2.86, 1 gained / 10 lost | thinking now closes (60/64), but `</think>` keeps firing in the answer section, which was never in the calibration |
| **m3c** | 0.3 · whitened "done" + 0.7 · whitened "late line end", answer section in the calibration | **30/64** | **+14.1 pts, SE 4.9, z 2.86, 10 gained / 1 lost** | — passes; medium 6 → 12, hard 0 → 3, easy 15 → 15; 0 stray closes |

## Confirmation (practice64 again, new random seed, both models)

| | original | m3c | difference | SE | z | gained / lost |
|---|---|---|---|---|---|---|
| seed 0 (pass 1 vs 10) | 21 | 30 | +14.1 pts | 4.9 | 2.86 | 10 / 1 |
| seed 1 (pass 11 vs 12) | 23 | 29 | +9.4 pts | 3.7 | 2.55 | 6 / 0 |
| pooled (128 answers) | 44 | 59 | +11.7 pts | 3.1 | 3.83 | 16 / 1 |

Same 64 problems in both seeds: this rules out a lucky sample, not a lucky problem set. The sealed 128 decide.

## Sealed test (passes 13–14, run once, 128 unseen problems)

| | easy (30) | medium (30) | hard (68) | all (128) | cut | wrong | mean tokens |
|---|---|---|---|---|---|---|---|
| original | 30 | 9 | 4 | **43 (33.6%)** | 85 | 0 | 12,765 |
| m3c | 30 | 19 | 18 | **67 (52.3%)** | 38 | 23 | 10,591 |

**Paired +18.8 points, SE 3.5, z 5.41; 24 gained, 0 lost. PASS.**

## Do-no-harm (passes 15–16) and 4-bit (passes 17–18)

| check | original | m3c |
|---|---|---|
| MMLU 200, thinking off | 52.5% | 52.5% (200/200 same answers) |
| perplexity change, prose / code | — | +0.00% / +0.03% |
| practice64 at MLX 4-bit | 17/64 | 25/64 (+12.5 pts, z 3.00, 8 / 0) |
