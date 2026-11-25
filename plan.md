# WeightTrain plan

Written 2026-11-14. Status 2026-11-21: **models 1 and 2 complete** (0.8B: sealed 13.3% → 52.3%; 2B: sealed
65.6% → 73.4%). Model 3 (MiniCPM5-2B, coding) started.

**End goal changed by the user on 2026-11-21: 3 models total, then write up** — a blog post, a publishable
APA-style paper, and a short social post. Models 4–5 move to future work. Everything else below still applies.

**End goal, agreed with the user on 2026-11-16 — this decides when work stops:**

Five local models, each with one locked edit that beats its own unedited version on a sealed test it never saw, by
at least 2 standard errors, passes the do-no-harm check, and is exported as a runnable MLX 4-bit model. One summary
page covers all five. A model where no edit wins within its limits is recorded as "no fix found" and closed.

Limits per model (whichever comes first ends work on it):
- 2 days from start to export.
- Reproduce the known fix if one exists, then **at most 3 edit attempts of my own**.
- 50 test passes.
- The sealed test runs **once**; its result is the result.

Must have (per model): one locked edit · sealed test result, paired against the base · do-no-harm table · MLX 4-bit
export that loads and answers · REPORT.md with graphs · disk cleaned of everything not needed for the above.

Later, not part of done: GGUF and Ollama builds · beating a published edit · combining edits · more variants of a
model that already passed · models 6–20.

Model 1 under these rules: edit attempts used 3 of 3 (A, M, M2). Locked edit: **M2** (user's pick, 2026-11-16;
ties the published edit over 160 problems and never ends without an answer).

Model 2 (Qwen3.5-2B) under these rules: 1 of 3 attempts (m2a = model 1's M2 recipe, layers 15/19), 13 of 50 passes.
Sealed +7.8 (3.14 SE); 4-bit edit vs 4-bit original +31.2 (5.0 SE). Details: `runs/qwen35-2b/REPORT.md`.

**Rules added by the user on 2026-11-14 — these override anything below that disagrees:**

1. **One model at a time.** Model 1 (Qwen3.5-0.8B) is finished end to end before model 2 is downloaded.
2. **No clutter.** Everything stays inside this folder. Test output is deleted when the test is done. An edit is
   saved as its changed columns only (0.26 MB per tensor for model 1). Answers are stored compressed. `data/` and
   `exports/` stay empty until they are needed.
3. **One results folder per model:** `runs/<name>/` with `REPORT.md` (results in plain language), `notes.md`
   (lab notebook), `graphs/` (SVG, made by `wsp/report.py`), `passes/` (every answer), `edits/` (every edit).
Current state of the work is always in `runs/<name>/notes.md`, not here.

Labels used below: **verified** = checked today by running or reading it. **expected** = a prediction.
**unverified** = recalled, must be checked at the step that uses it.

---

## 1. Goal, in one paragraph

Take small open-weight models, one at a time, and have Claude Code (Fable 5.2) improve each one by **editing its
weights directly** — no training, no gradient descent — the way the post in `weight-space-programming/` describes.
Start with 5 models whose full-precision download is under 8 GB. Each finished model ships as an edited checkpoint
plus a small, reproducible edit file, a report, and a quantized build that runs in Ollama.

### What "better" can honestly mean

- Evidence so far says weight edits **remove a defect that hides ability**. They have not been shown to add ability.
  - The post: Qwen3.5-0.8B went 7.9% → 48.1% on MATH-500 because a thinking loop was removed.
  - The post: Gemma 3 270M, which does not loop, gained nothing.
  - arXiv 2606.13705 (Gemma 4 loops): edits "do not add missing knowledge".
- So the claim we aim for per model is: **beats its own base on a held-out benchmark, by more than noise, with no
  regression elsewhere, and the gain survives 4-bit quantization.**
- A model where no fixable defect is found is recorded as such. That is a valid result, not a failure to hide.

### Two things that are unproven

- **Fable 5.2 has not been shown to do this.** The post used GPT-6 Astra. Model 1 exists to measure me.
- **Model 1 is not a blind test.** I have already read the published answer (which layers, how many neurons). It
  validates the tooling. The first blind test is model 2.

---

## 2. Rules of the game

Same rules the post gave its model, kept so results are comparable.

**Allowed**
- Any analysis of the weights (norms, SVD, products of matrices, projections onto token embeddings).
- Forward passes through a counted tool, within a budget.
- Any edit I compute in closed form and can justify with evidence.

**Not allowed**
- Gradient descent, backward passes, optimizers, or iteratively fitting weights to a loss.
- Editing the token embeddings, the output layer, or the final norm.
- Multiplying a whole tensor by a constant (a global knob, not a repair).
- Changing the tokenizer, chat template or sampling settings and calling it an improvement.
- Making the model skip or cut short its thinking regardless of the problem.
- Looking at the held-out test set before the edit is frozen.

**Open decision (not blocking):** the post's authors say the concept does not rule out gradients as a *reading* aid.
Default here: no gradients for all 5 models. Revisit only after model 2.

---

## 3. Machine and tooling

| Item | Value | Status |
|---|---|---|
| Mac | MacBook Pro M4 Max, 36 GB memory, 32-core GPU | verified |
| Free disk | 454 GB | verified |
| Installed | Ollama, `uv`, Python 3.12 | verified |
| Missing | MLX, mlx-lm, Hugging Face CLI, PyTorch | verified |
| Runtime for forward passes | mlx-lm 0.32.2 (current on PyPI) | verified version; behaviour unverified |
| Reference harness | `github.com/RikiyaT/lm_understand-logs` → `harness/`, `exp8_math/` | verified it exists; not yet read |

Why not Ollama for the work itself: it serves quantized models and exposes no weights or activations. Ollama is only
the final delivery format.

---

## 4. Folder layout (to be created in Phase 0)

```
WeightTrain/
  plan.md                      this file
  weight-space-programming/    the post, offline (done)
  reference/                   authors' harness + exp8_math, cloned read-only
  wsp/                         our toolkit (Python)
  models/<name>/               full-precision downloads, never modified
  data/<name>/practice/        problems I may look at
  data/<name>/holdout/         sealed until the edit is frozen
  runs/<name>/                 THE RESULTS FOLDER for one model
    REPORT.md                  results in plain language, with the graphs — open this first
    notes.md                   lab notebook; first thing read every session
    graphs/                    SVG graphs, made by wsp/report.py from the files below
    passes.jsonl               ledger of every forward pass (counts against budget)
    passes/gen_<k>_<label>/    every answer of pass k (answers.jsonl.gz), its summary and speed log
    edits/<edit-id>/           spec.json (what was asked), edit.json (what changed, hashes), changed columns only
    acts/                      saved activations (deleted when the model is finished)
  exports/<name>/              edited checkpoint, 4-bit MLX, GGUF — made only for an edit that passed its holdout
```

`models/` is read-only by convention. Every edit is an **overlay**: the changed tensors plus a manifest. The original
is never touched, so every edit is reversible by deleting a folder.

---

## 5. How Fable edits weights

### 5.1 The three ways I look at a model

| Kind | What happens | Cost | Counts against budget |
|---|---|---|---|
| **Static read** | I open the weight file with a memory map and compute on the numbers. No input is run. | Seconds | No |
| **Teacher-forced pass** | The model *reads* text I chose. I save what I asked for: residual stream per layer, MLP hidden activations, attention patterns, per-token log-probabilities. | Seconds to minutes | Yes, 1 pass |
| **Generation pass** | The model *answers* a batch of problems under the fixed protocol. Original and edited run on the same problems. I get accuracy, cut-off rate, and the paired difference. | Minutes to hours | Yes, 1 pass |

I work through scripts run from the terminal. Nothing lives in my head between sessions: findings go to `notes.md`,
every pass goes to `passes.jsonl`, every candidate edit is saved to disk before it is tested.

### 5.2 The toolkit (`wsp/`) — what each command does

1. `wsp inspect <model>` — list tensors, shapes, layer types; per-layer norms and spectra. Static.
2. `wsp probe <model> --texts ... --save resid,mlp,attn` — one teacher-forced pass, saves activations.
3. `wsp generate <model> --problems ... --both` — one generation pass, original vs edited, paired.
4. `wsp edit <model> --spec edit.json` — apply an edit spec to an overlay. Refuses out-of-scope tensors.
5. `wsp export <model> --edit <id>` — materialize the edited checkpoint, quantize, register in Ollama.

Where the authors' `harness/` code fits and its licence allows (listed as MIT / CC BY 4.0 — unverified until read),
adapt it instead of writing from scratch.

### 5.3 The edit operations (the whole vocabulary)

All are computed directly; none uses an optimizer.

| # | Operation | What it changes | Example from the post |
|---|---|---|---|
| 1 | **Neuron scale** | One column of an MLP output matrix × a factor. 0 = off, >1 = louder, negative = sign flip. | Probe: switch off neuron 11 of layer 19 |
| 2 | **Direction removal** | `W' = W − s · u (uᵀ W)` on chosen columns of an output matrix. Stops those neurons from writing direction `u`. `s` is the strength. | Final edit: 64 columns in each of layers 15 and 19 |
| 3 | **Head scale** | The output columns of one attention head × a factor. | 25-pass run: head 7 of layer 15 switched off |
| 4 | **Read-side change** | What a neuron or head *listens to* (input-side rows, or query/key weights). Changes when it fires. Higher risk; used last. | Not used in the post's final edits |

### 5.4 The loop I run on every model

1. **Diagnose.** Read failing outputs. Sort failures into categories. Go after the largest one I can explain.
2. **Build matched pairs.** Same problem, same opening, then two continuations: one healthy, one showing the
   failure (for loops: productive checking vs. the same sentence repeated). Wording is matched so the difference is
   the behaviour, not the vocabulary.
3. **Find the direction.** Teacher-force both. At each layer, `u` = normalized mean difference of the residual
   stream between failing and healthy continuations.
4. **Find the writers.** For each MLP neuron `j` in layer `l`: contribution = its activation × how much its output
   column points along `u`. Rank neurons by how much more they contribute in the failing case. Same for heads.
5. **Smallest probe first.** Switch off the single strongest writer. Run a paired generation pass.
6. **Widen only if the probe moved the right way.** One neuron → the direction in one layer → the direction in the
   layers that amplify it.
7. **Dose-response.** Try strengths 1, 2, 3, 6. A real mechanism responds in proportion. Back off when accuracy on
   already-correct problems starts to fall.
8. **Controls.** (a) A random direction of the same size must *not* reproduce the gain. (b) A neighbouring head or
   layer must *not* reproduce it.
9. **Selectivity check.** Teacher-forced: the log-probability of the failing continuation should drop; the
   healthy continuation should not.
10. **Minimize.** Keep the fewest columns that hold the gain.
11. **Freeze.** Write the edit file. No further changes after this point.

### 5.5 The edit file (what gets saved)

```json
{
  "base": {"repo": "Qwen/Qwen3.5-0.8B", "commit": "<sha>"},
  "edits": [
    {"tensor": "model.language_model.layers.19.mlp.down_proj.weight",
     "op": "remove_direction", "columns": [11, 204, "..."], "strength": 6,
     "direction_file": "u_layer19.npy",
     "sha256_before": "...", "sha256_after": "..."}
  ],
  "evidence": "runs/<name>/report.md"
}
```

Applying it is deterministic and checks the hashes, so anyone can rebuild the edited model from the base.

---

## 6. Phase 0 — one-time setup

Estimated: about 2 hours of my work, plus download time. Needs your OK because it installs packages and downloads
about 3.5 GB.

1. Create a Python environment in the project with `uv`. Install `mlx`, `mlx-lm`, `huggingface_hub`,
   `safetensors`, `numpy`. (expected commands; versions verified on PyPI)
2. `git init` the project, with `models/`, `exports/` and `acts/` ignored. Gives an undo history for notes and edits.
3. Clone `RikiyaT/lm_understand-logs` into `reference/`. Read `harness/` and `exp8_math/README.md` before running any of it.
4. Download `Qwen/Qwen3.5-0.8B` (1.75 GB) and the answer key `RikiyaT/Qwen3.5-0.8B-weight-space-programming` (1.75 GB).
5. Smoke test: load in mlx-lm, generate one answer with the model card's settings.
6. **Speed test (10 minutes).** Measure tokens/second, single and batched. Every time estimate in this plan is
   replaced with a measured one after this step.
7. Check that mlx-lm can apply a **presence penalty** (the post's protocol uses 1.5). If not, add a small logits
   processor. (unverified — this must match or model 1's numbers will not compare)
8. Build `wsp inspect`, `probe`, `generate`, `edit` and test them on model 1.

**Gate to leave Phase 0:** one generation pass and one teacher-forced pass both run end to end and write to the ledger.

**Progress, 2026-11-14:** steps 1–7 done. Step 8: `inspect`-style reading (`st.py`, `diff.py`), `generate.py`,
`grade.py`, `speed.py` built and tested; `probe` and `edit` not built yet.
- Step 7 finding: mlx-lm's presence penalty covers only the last 20 tokens and counts the prompt. `wsp/lm.py`
  replaces it with one over every generated token.
- The authors published prompts, the sandbox and the evaluation code, but **not** the forward-pass server. `wsp/` is
  written from scratch; only the grading rules are taken from their code (MIT, credited in `wsp/grade.py`).

---

## 7. The 14 steps, per model

Each step has one output. A step is not done until its output exists.

| # | Step | Output | Stop here if |
|---|---|---|---|
| 1 | **Download and pin.** Official repo, exact commit, checksum. | `models/<name>/`, commit in `notes.md` | Licence or gating blocks use |
| 2 | **Read the model card.** Sampling settings, chat template, known issues. | "Protocol" section in `notes.md` | — |
| 3 | **Load and verify.** Runs in mlx-lm; one sane answer; tokens/second measured. | Speed line in `notes.md` | It cannot be run correctly |
| 4 | **Pick the target area and metric.** One benchmark for the area; split into practice and sealed holdout. | `data/<name>/` | — |
| 5 | **Baseline pass.** Base model on ~96 practice problems. Accuracy, cut-off rate, raw outputs. | Pass 1 in the ledger | — |
| 6 | **Failure taxonomy.** Read the failures; count by category (loop/cut-off, format, tool call, arithmetic, missing knowledge, refusal). | Table in `notes.md` | **No mechanical category above ~20% of failures → record "no fixable defect found" and move on** |
| 7 | **Static read.** Layer map, norms, which MLP columns write toward which tokens. | Findings in `notes.md` | — |
| 8 | **Localize.** Matched pairs → direction → ranked writers (§5.4 steps 2–4). | `acts/`, ranked list | No stable direction across pairs |
| 9 | **Probe and widen.** Smallest edit first, then wider, each with a paired pass (§5.4 steps 5–6). | Candidate edits on disk | Three probes in a row show no effect |
| 10 | **Dose-response and controls.** (§5.4 steps 7–9). | Table: strength vs. gain; control results | The random control matches the gain |
| 11 | **Minimize and freeze.** Fewest weights that keep the gain. | `edit.json`, frozen | — |
| 12 | **Sealed holdout, once.** Base vs. edited, paired, on problems never used. | Result in `report.md` | Gain < 2 standard errors → report as "not established" |
| 13 | **Do-no-harm suite + quantized check.** A fixed cross-area set, base vs. edited. Then repeat a 128-problem check on the 4-bit build. | Two tables in `report.md` | A regression beyond 2 standard errors → back to step 10 with lower strength |
| 14 | **Export and write up.** Edited checkpoint, 4-bit MLX, GGUF, Ollama tag `<name>-wsp`, `report.md`. | `exports/<name>/` | — |

**Budget per model:** 50 forward passes (the post's largest budget). Failed passes do not count.
**Session rule:** every session starts by reading `notes.md` and `passes.jsonl`, and ends by updating them.

---

## 8. Measurement rules

- **Fixed protocol.** Sampling settings come from the model card and never change between base and edited.
- **Paired comparison.** Base and edited answer the *same* problems. Report the paired difference and its standard error.
- **Noise.** Accuracy over `n` answers has a standard error of about `50/√n` points: 4.4 points at 128, 2.2 at 512.
  A gain counts only if it clears 2 standard errors on problems not used to choose the edit.
- **Judge by accuracy, not by cut-off rate.** Fewer cut-offs with no more correct answers is not an improvement.
- **Not an early exit.** For rescued answers, check the thinking is still substantial (the post: median ~9,000 tokens).
- **Holdout is sealed.** Run once, after the edit is frozen. If I look and then change the edit, the holdout is burned
  and a new one must be drawn.

### Making generation passes affordable locally

The post's protocol allows 32,768 tokens per answer. That is slow on a laptop. Two levers:

1. **Loop detector.** Stop an answer once it repeats verbatim past a threshold; count it as a cut-off.
   **Must be validated first** on model 1: run ~32 answers to full length and confirm the detector's calls match
   real cut-offs. (The post found cut-off answers were ~95% duplicated phrases in their tails.)
2. **Shorter cap for iteration** (8,192 tokens), full cap only for the final holdout.

expected: with both levers, a 64-problem pass on the 0.8B model takes minutes, not hours. Unverified until the speed test.

---

## 9. The first five models

All verified today on the Hugging Face API: public, not gated, sizes as listed.

| # | Repo | Download | Layers × width | Runs in mlx-lm | Area | Licence |
|---|---|---|---|---|---|---|
| 1 | `Qwen/Qwen3.5-0.8B` | 1.75 GB | 24 × 1024 | yes (`qwen3_5`) | Calibration (math reasoning) | Apache 2.0 |
| 2 | `Qwen/Qwen3.5-2B` | 4.55 GB | 24 × 2048 | yes (`qwen3_5`) | General reasoning | Apache 2.0 |
| 3 | `openbmb/MiniCPM5-2B` | 5.03 GB | 42 × 2048 | yes (`llama`) | Coding | Apache 2.0 |
| 4 | `ibm-granite/granite-4.2-3b` | 7.32 GB | 40 × 2560 | yes (`granite`) | General / tools | Apache 2.0 |
| 5 | `IFM/K2-Horizon-0.9B` | 2.16 GB | 28 × 1536 | **no** — custom code | General (tiny) | Apache 2.0 |

Total: about 21 GB, plus 1.75 GB for the model 1 answer key.

### Model 1 — Qwen3.5-0.8B (tooling validation, not blind)

- **Target:** thinking loops on math problems. Protocol from the post: thinking on, temperature 0.6, top-p 0.95,
  top-k 20, presence penalty 1.5.
- **Data:** the authors' own practice files in `exp8_math/practice/`. Holdout: MATH-500 and GSM8K test.
- **Four stages, each a gate:**
  1. **Reproduce the failure.** Base model should hit the token limit on roughly 90% of answers (post: 93%).
  2. **Apply the published edit.** Should land near the post's 48.1% / the independent re-run's 51.0% on MATH-500.
     If stages 1–2 do not reproduce, the bug is in our harness. Fix before anything else.
  3. **Re-derive it.** Run §5.4 with my own tools. Compare my direction and my top-64 neurons with the published ones.
  4. **Try to beat it.** A third of edited answers still hit the limit. The post's own notes point at two copying
     heads (layer 15 head 7, layer 19 head 5) as a second site.
- **What model 1 tells us:** whether the toolkit is correct (stages 1–2) and whether I can localize a known
  mechanism (stage 3). Only stage 4 could produce something new.

### Model 2 — Qwen3.5-2B (first blind test)

- **Target:** unknown until measured. The 0.8B model card says the 0.8B loops *more* than its siblings, so the 2B
  may not have the same defect.
- **Plan:** baseline pass on the same math practice set. If cut-offs dominate, run the model 1 recipe and see
  whether it transfers. If not, the taxonomy at step 6 picks the target.
- **Why second:** same architecture as model 1, so the tooling is already proven; the question is only whether I
  can find something nobody has published.

### Model 3 — MiniCPM5-2B (coding)

- **Reported profile** (vendor numbers, secondary sources): strong on short code — LiveCodeBench v6 69.1 — but weak
  on long tasks — SWE-bench Pro 14.4 vs. 28.2 for Qwen3.5-4B; Terminal-Bench 8.6 vs. 25.8.
- **Target candidates:** malformed or abandoned tool calls in multi-step work; stopping early; repetition in long outputs.
- **Metric:** a small local multi-step coding task set with unit tests, plus a LiveCodeBench-style subset as the
  regression check. Dataset choices are unverified until step 4.
- **Needs your OK before its first run:** this model's output is *code that gets executed*. It will run in a
  sandbox with no network, a time limit, and writes confined to a temp folder.

### Model 4 — Granite 4.2 3B (general / tools)

- **Target:** unknown. I have no failure data for it; ranks 3rd among models of 4B and under on Artificial
  Analysis (191). Its repo ships a thinking parser, so it has a reasoning mode.
- **Plan:** read the card, then let the step 6 taxonomy choose between reasoning, structured output and tool calls.

### Model 5 — K2 Horizon 0.9B (tiny, fully open)

- **Friction:** not supported by mlx-lm; the repo ships its own model code. It would run through PyTorch on the
  Mac's GPU, and I read `modeling_k2_horizon.py` before executing it.
- **Why keep it:** released with training data and recipe, so claims about what it learned can be checked.
- **Fallback if the friction is too high:** `LiquidAI/LFM2.5-2.6B` (5.39 GB, supported by mlx-lm). Its licence is
  listed as "other" and must be read first.

### Not in the first five — needs your decision

There is **no writing model** under 8 GB at full precision. If "under 8 GB" may mean *quantized size for running*,
these come back, and all still fit in 36 GB for editing:

| Repo | Full precision | Area | Why |
|---|---|---|---|
| `google/gemma-4-E4B-it` | 16.0 GB | Writing / general | Documented repetition loops; a paper fixed them with a few neurons |
| `Qwen/Qwen3.5-9B` | 19.3 GB | General / agents | Documented math loops and malformed tool calls |
| `ornith-ai/Ornith-1.0-9B` | 18.8 GB | Coding | Highest claimed SWE-bench Verified under 10B (69.4) |
| `IFM/K2-Horizon-7B` | 18.0 GB | General | Ranks above Gemma 4 31B on Artificial Analysis's small tier |

---

## 10. Time and cost

- **Nothing here costs money** beyond normal Claude Code usage. All models are free downloads; all compute is local.
- **Measured 2026-11-14 on model 1 (0.8B):** 137–170 tokens/s for one answer; 1,186 tokens/s total at batch 32;
  1,439 at batch 64. Speed far into a long answer is not measured yet.
- expected from those numbers: one 96-problem pass at the full 32,768-token cap ≈ 3M tokens ≈ 40–60 minutes. A
  50-pass budget is then 1.5–3 days of machine time for model 1. Larger models are slower roughly in proportion to size.
- The Mac will be busy and warm during generation passes. Passes can run overnight; they need no input from you.

---

## 11. Risks

| Risk | What I do about it |
|---|---|
| I cannot localize mechanisms as well as the post's model | Model 1 stage 3 measures this before more time is spent |
| A gain is noise | Paired runs, 2-standard-error rule, sealed holdout, random-direction control |
| An edit helps one benchmark and hurts everything else | Do-no-harm suite at step 13 |
| The gain disappears after quantization | 4-bit check at step 13, before export |
| Sampling differs from the post (presence penalty) | Checked in Phase 0 step 7 |
| Work is lost between sessions | `notes.md` + ledger + edits on disk; git history |
| Model-generated code runs on your Mac (model 3) | Sandbox; your explicit OK first |
| A repo ships executable code (model 5) | Read it before running; fallback model named |
| Benchmark numbers quoted here are vendor-reported | Every model gets its own local baseline at step 5; nothing depends on quoted scores |

---

## 12. What happens next

1. You approve this plan, or change it.
2. You answer the one open decision: keep the strict under-8-GB list, or allow the four larger models in §9.
3. I run Phase 0 (§6) — it installs packages and downloads 3.5 GB, so it waits for your OK.
4. Model 1, stage 1.
