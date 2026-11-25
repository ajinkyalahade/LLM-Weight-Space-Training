"""Grading a response to a math problem, and noticing a response that ends in a loop.

The rules are those of the post's evaluation, so that "correct" means the same here. The answer extraction
(answer_text, _box_at, last_boxed) and the loop test (verbatim_loop_end) are taken from
reference/lm_understand-logs/exp8_math/code/mathqa.py (MIT License, copyright 2026 the authors of "LLMs can understand
and write neural network weights"; its box reading follows hendrycks/math, MIT). The checker is math-verify 0.9.0.
"""
import functools

import math_verify


def answer_text(text):
    """The text the answer is read from: what follows the last '</think>' (the whole text if there is none)."""
    return text.split("</think>")[-1].strip() if "</think>" in text else text


def _box_at(text, idx):
    key = "\\boxed" if text.startswith("\\boxed", idx) else "\\fbox"
    rest = text[idx + len(key):]
    r = rest.lstrip()
    if r.startswith("{"):
        depth = 0
        for k, ch in enumerate(r):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return r[1:k].strip() or None
        return None
    if key == "\\boxed" and rest.startswith(" "):
        return rest.strip().split("$")[0].split("\n")[0].strip() or None
    return None


def last_boxed(text):
    """The content of the last \\boxed{..} (else the last \\fbox{..}) of a text, or None."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    return _box_at(text, idx)


@functools.lru_cache(maxsize=1 << 16)
def _parsed(s):
    return math_verify.parse("\\boxed{" + s + "}", parsing_timeout=5)


def is_correct(pred, gold):
    if pred is None or gold is None:
        return False
    g, p = _parsed(gold), _parsed(pred)
    if not g or not p:
        return False
    try:
        return bool(math_verify.verify(g, p, timeout_seconds=5))
    except Exception:  # noqa: BLE001
        return False


def verbatim_loop_end(text, max_period=200, min_repeats=3, min_span=40):
    """True if the text, without its last word, ends with one block of 1..200 words repeated in a row at least 3 times
    and over at least 40 words."""
    w = text.split()[:-1]
    n = len(w)
    for p in range(1, max_period + 1):
        reps = max(min_repeats, -(-min_span // p))
        if reps * p > n:
            break
        blk = w[n - p:]
        if all(w[n - (j + 1) * p:n - j * p] == blk for j in range(1, reps)):
            return True
    return False


def grade(text, gold, finish):
    """One response -> its record. A response cut at the token limit is a failure whatever it contains, as in the
    post's forward-pass tool (its failures are: wrong answer, finished without an answer, cut at the token limit)."""
    cut = finish == "length"
    pred = last_boxed(answer_text(text))
    ok = (not cut) and is_correct(pred, gold)
    outcome = "correct" if ok else "cut_at_max_tokens" if cut else "no_answer_finished" if pred is None else "wrong_answer"
    return {"pred": pred if pred is None else pred[:200], "correct": ok, "outcome": outcome,
            "closed_thinking": "</think>" in text, "loop_end": verbatim_loop_end(text)}
