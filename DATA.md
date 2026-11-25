# Data and files not in this repository

This repository holds the code (`wsp/`), the plan (`plan.md`), each model's report, lab notes, pass ledger and graphs
(`runs/<model>/`), the edit files themselves (`runs/<model>/edits/`), and the write-up (`writeup/`).

The files below were left out because they are large, can be rebuilt, or belong to someone else. If you want to
review any of them, contact the author: **[contact: add how people should reach you]**.

## Available from the author on request

| What | Where it lives locally | Size | Why it is not here |
|---|---|---|---|
| Edited models: full-precision and MLX 4-bit, all three models | `exports/` | 13 GB | Too large for git |
| Every model answer of every pass (text, grade, outcome) | `runs/*/passes/*/answers.jsonl.gz` | 25 MB | Raw data; summaries and ledgers are included |
| Measured activations and directions per pass | `runs/*/passes/*/*.npz` | 45 MB | Intermediate data; the resulting edits are included |
| LiveCodeBench problems with their test cases, as used | `data/minicpm5-2b/lcb/problems.jsonl` | 1.1 GB | Third-party dataset; rebuild it as below |

`writeup/figures/make_figures.py` needs the answer files to regenerate the figures; `writeup/figures/facts.json`
holds every number it computed.

## Available from the original source

| What | Source |
|---|---|
| Base models | Hugging Face: `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B`, `openbmb/MiniCPM5-2B` (revision `42433e90dad95123ab87729fd7011e7110629556`) |
| LiveCodeBench problems | Hugging Face dataset `livecodebench/code_generation_lite`, files `test5.jsonl` and `test6.jsonl`; convert with `python -I wsp/lcb.py` |
| The blog post this work follows | Takehi, Teoh, Zhang & Khattab (2026), https://rikiyat.github.io/blog/weight-space-programming/ |
| The original authors' experiment logs | https://github.com/RikiyaT/lm_understand-logs |
| Prose text used in the perplexity check (`data/qwen35-0.8b/harm/prose.txt`) | The text of the blog post above |
