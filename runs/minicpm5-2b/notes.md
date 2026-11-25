# MiniCPM5-2B — lab notebook

## Fixed facts

- Model: `openbmb/MiniCPM5-2B`, revision `42433e90dad95123ab87729fd7011e7110629556`, downloaded 2026-11-21 to
  `models/MiniCPM5-2B` (4.7 GB, one safetensors file, bf16). Repo has no code files. Apache 2.0.
- `llama` architecture: 42 layers, width 2048, 6,144 MLP neurons per layer, 16 heads / 2 KV heads, untied embeddings,
  vocab 130,560. Chat template with `enable_thinking`.
- Vendor sampling: temperature 1.0, top-p 0.95, min-p 0. Vendor's own note: "If you encounter repetitive outputs,
  try ... repetition_penalty=1.05" — looping is a documented weakness.
- Load check: `mlx_lm generate` writes a correct `is_prime` after a short think; 73 tok/s single stream; peak 5.1 GB.

## Limits (end goal, top of `plan.md`)

- Started 2026-11-21. **Time limit: 2026-11-23, same hour.** At most 3 edit attempts. 50 passes. Sealed test once.

## Proposed plan (waiting for the user's OK: model-written code gets executed)

- Data: LiveCodeBench `code_generation_lite` (`test5.jsonl` + `test6.jsonl`, ~690 MB), stdin/stdout tests.
  Split by random seed into practice 64 / collection / sealed 128, fixed before any run.
- Private tests are stored as base64 → zlib → pickle of a string. Load with a restricted unpickler that refuses
  every class (a pickled string needs none), so loading cannot run code.
- Sandbox for running the model's code: macOS `sandbox-exec` profile denying network and all file writes except a
  fresh temp folder; `python -I`; CPU-time and memory limits; wall-clock timeout per test; temp folder deleted after.
- Baseline: thinking on, vendor sampling, cap 16,384, seed 0. Then pick the target from what fails (loops expected).

## Log

- **2026-11-21** Model 2 closed (user OK), `models/Qwen3.5-2B` deleted. Model 3 downloaded and load-checked.
- User: "you pick and continue" → coding test in the sandbox. User also changed the end goal: **3 models total, then
  blog post + APA paper + short social post** (plan.md top, memory).
- Data: `livecodebench/code_generation_lite` rev `0fe84c39…`, `test5.jsonl` + `test6.jsonl` (660 MB) →
  `wsp/lcb.py` (class-refusing unpickler) → `data/minicpm5-2b/lcb/problems.jsonl`: **217 stdin problems, all
  AtCoder** (50 easy / 50 medium / 117 hard), 2024-09-22 → 2025-04-06, median 43 tests; 125 LeetCode-style
  (functional tests) left out.
- Splits (random.Random(7), by difficulty, before any run): `data/minicpm5-2b/lcb/sealed128.json` (30/30/68),
  `problems_practice64.json` (15/15/34), `problems_collect25.json` (5/5/15). Disjoint.
- `wsp/sandbox.py`: sandbox-exec (no network, no reads under /Users, writes only in a fresh temp dir, no exec except
  system python, CPU 8 s, wall 6 s per test, 60 s per answer, 64 MB files). **Selftest 8/8 passed** (network, home
  read, outside write, exec all blocked). LiveCodeBench grading: last ```python block after thinking, all tests must
  pass, stop at first failure.
- Protocol `minicpm_code` in `lm.py` (thinking on, temp 1.0, top-p 0.95, no penalty, cap 16,384). Trial (no ledger,
  deleted): 4 easy problems, 4/4 correct, all tests passed, 62 s.
- **Pass 1 baseline** on practice64 started.
- **Pass 1 result: 21/64 (32.8%).** Easy 15/15, medium 6/15, hard 0/34. 42 cut at 16,384 (8 medium, 34 hard),
  1 wrong answer, 0 no-code. 60 min, 223 tok/s, peak 23.6 GB.
- **The failure is not looping, it is never stopping.** All 42 cut answers are still thinking at the cap (0 closed
  `</think>`); 0 end in a verbatim loop; repeated-8-gram share in the last 2,000 tokens is < 0.2 in 41 of 42 (max
  0.22); no 1,000-token window anywhere reaches 0.8. The tails are endless re-verification ("Now scenario: s=…
  Works."). Correct answers used 618–12,300 tokens. So model 2's loop recipe does not apply as is.
- Before choosing a target: **pass 2, forced close** (`wsp/close.py`, new): each cut answer's thinking cut at 8,192
  and at 16,384 tokens (back to a paragraph break), `\n</think>\n\n` appended, up to 4,096 answer tokens, graded in
  the sandbox. This is the ceiling of a "stop and answer" edit.
- **Pass 2 result.** Closed at 16,384: **10 of 42 now correct** (7 of the 8 cut medium, 3 hard). Closed at 8,192:
  6 (all 6 also in the 16k set). So 21 → up to 31 of 64 if the model stopped by itself near the cap; stopping
  earlier loses some. 20–24 forced answers ran past the 4,096-token answer window (they write a long "Solution
  Explanation" first) — a limit of this diagnostic, not a new failure. Wrong: 15 wrong output, 5 crash, 4 timeout.
- **Target: stop thinking and answer.** Attempt 1 plan: a one-row edit. Learn a "done" direction d in the final
  hidden state (mean at the line end right before `</think>` minus mean at other thinking line ends) and add
  beta·d to the `</think>` row of `lm_head` (token 9): its logit rises by beta·(h·d), mostly where the state looks
  finished. `wsp/closedir.py` (measure / make); `lm.load` now also reads row edits.
- Data hygiene: all 217 problems are assigned, so practice64 is the only decision set besides sealed; d must not be
  fitted on it. **Pass 3: collect25 × 2 samples** (seed 1) to fit d on.
- Pass 3 result: 8/50 (all easy), 42 cut, 0 loops. Only 8 answers close → 9 closing positions (one cut answer
  also closed once, then kept going).
- Pass 4 (`closedir.py measure`, forward): 21,183 thinking line ends. d = mean at closing − mean elsewhere.
  h·d: closing 144 ± 9, other line ends −28 ± 56. Held out (d refit without the answer): closing 120–154, the
  answer's other line ends median ≈ −5. Cut answers after 8k: median −13, 99th pct 108. Simulated (teacher-forced)
  at beta 0.12: 27% of cut answers close by 12k.
- **Attempt 1 = m3a** (beta 0.12; `</think>` row norm 0.780 → 0.809). **FAILED at once.** Pass 5 on practice64:
  after 30 s, 12 of the first 13 easy answers had no code — `</think>` emitted mid-sentence from the first tokens
  ("We need</think></think>We need</think>…"). Stopped after 13 answers (1 correct). Cause: d was calibrated only
  at line ends; positions inside a line and at the very start of the thinking were never measured, and they score
  high on d. Edit attempts used: **1 of 3**.
- Pass 6 (forward, collect answers again): all line ends + first 256 thinking positions + random 10% of the rest
  (weight 10) — 101,053 positions. Simulated (teacher-forced; expected closes = Σ weight·p):
  - plain mean difference: beta 0.10 → 10% of answers close mid-sentence somewhere, 94% of cut close by 12k;
    beta 0.12 (≈ m3a) → 44% mid-sentence. Confirms the m3a failure.
  - whitened (LDA, shrink 0.1): closing 49 ± 6 vs others −2.3 ± 1.6, mid-line max 9.5: very specific, but cut
    answers never look "done" (late line ends 99th pct 1.2) → only 2–4% of them close at any beta ≤ 1. The cut
    answers have not concluded; forced closing works because it makes the model write code from partial work.
- **Length is linearly readable.** Late direction (whitened, line ends ≥ 10k vs every other position), 5-fold held
  out by answer: median h·d_late at cut-answer line ends rises 3.1 (0–2k) → 7.5 (4–6k) → 11.6 (10–12k) → 14.1
  (14k+), corr with position 0.69; mid-line ≈ 1.4; early line ends ≈ 5.6. (`scratchpad/m3_late.py`)
- **Attempt 2 = m3b**: `</think>` row += 0.3·d_done + 1.0·d_late (`closedir.py make --shrink 0.1 --late-from
  10000 --use 0.3 --use-late 1.0`; row norm 0.780 → 1.404). Simulated: mid-sentence 0.1%, cut answers closed by
  8k / 10k / 12k / 14k = 30 / 49 / 72 / 89%. Late 1.2 would close 65% by 8k (too early: pass 2 rescued 6 at 8k vs
  10 at 16k). **Pass 7**: practice64, seed 0. First 5 answers (easy): all correct, close at sensible line ends.
  Side effect seen: stray `</think>` after the code (line after code also looks "done"); grading stays as fixed
  before the run (answer = text after the last `</think>`), so where that leaves no code it counts against m3b.
- **Pass 7 result: m3b FAILED.** 12/64 vs 21 (easy 15 → 10, medium 6 → 2, hard 0 → 0); −14.1 points, SE 4.9,
  z −2.86; gained 1, lost 10. The closing itself worked: 60/64 closed thinking (pass 1: 22). The answer section
  broke: 57/64 have stray `</think>`, 16 end with no code after the last one, 23 of 32 cut answers end in a loop
  of `</think>\n\n`. Diagnostic re-grade with strays deleted (not a score): still 12 — the strays derailed the
  generation, not only the grading. Cause: same class as m3a — the answer section after `</think>` (late, line
  ends) was never in the calibration, and it looks "late". Edit attempts used: **2 of 3**.
- **Attempt 3 plan (last) = m3c**: m3b with the answer section in the "other" class. Data from collect only:
  **pass 8** = `close.py` on pass 3 at K 10,240 / 14,336 (full texts now saved), then measure with
  `--answer-region` (forced closes contribute only their answer section), then pick strengths with
  stray-close ≈ 0. `make` regression-checked: reproduces m3b's simulation exactly on the pass-6 file.
- Pass 8 was frozen 17:48 → 19:21: the Mac ran on battery to 1% and hibernated (pmset log). No data lost; the
  pass resumed after wake. Now on AC. ~1.5 h of the 2-day limit lost.
- **Pass 8 result** (collect, forced close): **10 of 41** cut answers correct at K 10,240 and also 10 at 14,336
  (medium 6 / 7). Same ceiling as practice (pass 2: 10 of 42 at 16k) on an independent set.
- **Pass 9**: measure pass 3 + pass 8 with `--answer-region`: 132 answers, 160,445 positions, 59,420 in answer
  sections. With the answer section in the "other" class the late direction is less length-selective (thinking line
  ends 7.2 at 0–2k → 16.8 at 14k+; m3b's: 3.4 → 13.7), answer section median 1.8 (99.9th pct 9.8).
  Simulated (done 0.3 + late): late 1.0 → 92% of cut close by 8k (too early); **late 0.7 → stray in answer 1.6%,
  mid-sentence 0%, cut closed by 8k / 10k / 12k / 14k = 32 / 47 / 66 / 82%** (≈ m3b's timing); late 0.5 → only
  16% by 12k. Later start (12k) only shifts the same curve.
- **Attempt 3 = m3c** (last): `</think>` row += 0.3·d_done + 0.7·d_late (late from 10k, shrink 0.1, pass-9
  features); row norm 0.780 → 1.182. **Pass 10**: practice64, seed 0. First 5 answers (easy): all correct, each
  exactly one `</think>`. (Launched with `&` by mistake, so not a tracked task; a watcher waits on its pid.)
- **Pass 10 result: m3c PASSES on practice64. 30/64 vs 21 (46.9% vs 32.8%); +14.1 points, SE 4.9, z 2.86;
  gained 10, lost 1.** Easy 15 → 15, medium 6 → 12, hard 0 → 3. 52/64 close thinking (orig 22), 0 stray
  `</think>`, 0 loop ends; cut 42 → 26; wrong 1 → 8 (4 wrong output, 3 crash, 1 timeout). Lost: abc390_c (medium,
  cut). Gained: 7 medium + 3 hard; 8 of the 10 are problems the forced close rescued in pass 2.
  Strengths were chosen on the collect set only, so practice64 was a held-out test of one setting.
- No unused problems are left for a fresh confirmation set (all 217 assigned). Confirmation instead: **passes
  11–12 = original and m3c on practice64 with seed 1** (new samples, same problems), pooled with seed 0.
- **Passes 11–12 result: the gain replicates.** Seed 1: original 23/64, m3c 29/64; +9.4 points, SE 3.7, z 2.55;
  gained 6, lost 0 (easy 15 → 15, medium 7 → 12, hard 1 → 2). m3c closes thinking in 51/64 (orig 24); 1 answer
  with a stray `</think>`; 25 cut, 10 wrong. **Pooled seeds 0+1 (128 paired answers): 44 → 59, +11.7 points,
  SE 3.1, z 3.83, gained 16, lost 1.** Same problems both seeds, so this confirms the sampling noise, not new
  problems; the sealed 128 are the real test.
- **2026-11-22 LOCKED: edit m3c** (user: "ok"). `edits/m3c/tensors/lm_head.weight.npz` sha256
  `ae12512ef54dbaa06219e8db2a85cae8f2d201a11954f4875393bac4f96ff831`. Sealed test started, run once:
  `data/minicpm5-2b/lcb/sealed128.json` (30/30/68), original then m3c, protocol `minicpm_code`, cap 16,384,
  seed 0, batch 16. Pass rule fixed before: m3c − original ≥ 2 SE (paired).
- **2026-11-22 SEALED RESULT (passes 13–14, 120 + 96 min): PASS.** Original **43/128 (33.6%)** (easy 30/30, medium
  9/30, hard 4/68; 85 cut, 44 closed thinking; mean 12,765 tokens), m3c **67/128 (52.3%)** (easy 30, medium 19,
  hard 18; 38 cut, 23 wrong, 115 closed, 3 with a stray `</think>`; mean 10,591 tokens). **Paired +18.8 points
  (SE 3.5, z 5.41), 24 gained / 0 lost.** Larger than on practice (+11.7 pooled), mostly from hard (4 → 18).
- **Do-no-harm (passes 15–16, 19 s each, model 1's files):** MMLU 200 (thinking off) 52.5% → 52.5%, same answer on
  200/200; prose perplexity +0.00%, code +0.03%. No harm (expected: only the `</think>` row changed).
- Export: `exports/minicpm5-2b/m3c-bf16` (st.patch, 2,043 of 2,048 values changed in bf16; logits identical to the
  overlay, max diff 0.0) → `mlx_lm convert -q --q-bits 4` → `m3c-mlx-4bit` (1.3 GB, 4.501 bpw). Unedited 4-bit
  conversion in the scratchpad. **Passes 17–18**: unedited 4-bit and m3c 4-bit on practice64, seed 0.
- **Passes 17–18 result (58 + 49 min):** unedited 4-bit **17/64** (easy 14, medium 2, hard 1; 47 cut, 0 loops),
  m3c 4-bit **25/64** (easy 15, medium 7, hard 3; 30 cut, 8 wrong, 1 no answer; 48 closed; **11 with a stray
  `</think>`**, vs 0 at bf16). **m3c 4-bit vs unedited 4-bit: +12.5 points (SE 4.2, z 3.00), 8 gained / 0 lost.**
  Quantization costs both about the same (orig 21 → 17, m3c 30 → 25, each z ≈ −1.7); the 4-bit export still beats
  even the unedited bf16 model (25 vs 21). Temporary unedited conversion deleted. Graphs: `graphs/sealed_m3c_*`,
  `graphs/practice_4bit_*` (`report.py` now names the coding sources Easy / Medium / Hard).
- **MODEL 3 COMPLETE (2026-11-22).** All must-haves met: locked edit m3c (one row, 2,048 values), sealed +18.8
  (5.41 SE), do-no-harm passed, MLX 4-bit export beats unedited 4-bit (+12.5, 3.0 SE). Recommended to run:
  `exports/minicpm5-2b/m3c-bf16` (4-bit adds stray `</think>` in 11/64). 18 of 50 passes, 3 of 3 attempts,
  ~27 h of 48.
- Cleanup (user: "yes delete"): `models/MiniCPM5-2B` (4.7 GB) and `data/minicpm5-2b/lcb/raw` (660 MB) deleted. Kept:
  exports, `problems.jsonl`, splits, all passes and edits.
