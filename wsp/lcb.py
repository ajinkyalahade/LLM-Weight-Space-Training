"""LiveCodeBench problems -> a problems file for wsp/generate.py.

    python -I wsp/lcb.py --raw data/minicpm5-2b/lcb/raw/test5.jsonl data/minicpm5-2b/lcb/raw/test6.jsonl \
        --out data/minicpm5-2b/lcb/problems.jsonl

Keeps problems whose tests are all stdin/stdout (AtCoder, Codeforces); LeetCode-style "functional" problems need a
different harness and are left out. Each output row: id, source ("lcb_<difficulty>"), problem (the prompt, in
LiveCodeBench's own wording for stdin problems), answer (null: code is graded by its tests), tests, date, platform.

The private tests are stored as base64 -> zlib -> pickle of a JSON string. Unpickling untrusted data can run code, so
they are read with an unpickler that refuses every class: a pickled string needs none, so anything else fails.
"""
import argparse
import base64
import io
import json
import pickle
import zlib

PROMPT = ("You will be given a question (problem specification) and will generate a correct Python program that "
          "matches the specification and passes all tests.\n\nQuestion: {q}\n\nRead the inputs from stdin solve the "
          "problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code "
          "within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the "
          "algorithm and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```")


class NoClasses(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refused: {module}.{name}")


def private_tests(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        raw = zlib.decompress(base64.b64decode(s.encode("ascii")))
        val = NoClasses(io.BytesIO(raw)).load()
        assert isinstance(val, str), type(val)
        return json.loads(val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kept = skipped = 0
    with open(a.out, "w") as f:
        for path in a.raw:
            for line in open(path):
                r = json.loads(line)
                tests = json.loads(r["public_test_cases"]) + private_tests(r["private_test_cases"])
                if not tests or any(t["testtype"] != "stdin" for t in tests):
                    skipped += 1
                    continue
                f.write(json.dumps({
                    "id": f"lcb_{r['question_id']}", "source": f"lcb_{r['difficulty']}",
                    "problem": PROMPT.format(q=r["question_content"]), "answer": None,
                    "tests": [{"input": t["input"], "output": t["output"]} for t in tests],
                    "date": r["contest_date"][:10], "platform": r["platform"]}) + "\n")
                kept += 1
    print(f"kept {kept} stdin problems, skipped {skipped} (functional tests)")


if __name__ == "__main__":
    main()
