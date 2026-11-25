"""Run a model-written Python program against stdin/stdout tests, inside a macOS sandbox.

    grade_code(text, tests, finish) -> record with the same keys as grade.grade (plus tests_passed, tests_total, fail)
    python -I wsp/sandbox.py --selftest

The sandbox (sandbox-exec profile below), per test:
  - no network;
  - no file writes except inside a fresh temporary folder, which is deleted afterwards;
  - no reading anything under /Users (the user's files), except that temporary folder;
  - no starting other programs except the system Python itself;
  - CPU time limit, wall-clock timeout, file-size limit; the program runs as `python3 -I -B` with only the standard
    library (not this project's virtual environment).
Grading follows LiveCodeBench: an answer is correct only if it passes every test; the code is the last ```python
block after the thinking; outputs are compared line by line after stripping whitespace, with a float tolerance.
Testing stops at the first failing test.
"""
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grade  # noqa: E402

PY = shutil.which("python3", path="/opt/homebrew/bin:/usr/bin")
PY_REAL = os.path.realpath(PY)
TIMEOUT_S, CPU_S, FILE_BYTES, TOTAL_S = 6, 8, 64 * 1024 * 1024, 60

PROFILE = """(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{tmp}") (literal "/dev/null"))
(deny file-read* (subpath "/Users"))
(allow file-read* (subpath "{tmp}"))
(deny process-exec*)
(allow process-exec* (literal "{py}") (literal "{py_real}") (subpath "{py_root}"))
"""


def extract_code(text):
    body = grade.answer_text(text)
    blocks = re.findall(r"```(?:python|py|Python)?\s*\n(.*?)```", body, re.S)
    return blocks[-1] if blocks else None


def _same(out, exp):
    a = [x.strip() for x in out.strip().splitlines()]
    b = [x.strip() for x in exp.strip().splitlines()]
    if a == b:
        return True
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        xs, ys = x.split(), y.split()
        if len(xs) != len(ys):
            return False
        try:
            if any(abs(float(p) - float(q)) > 1e-6 * max(1.0, abs(float(q))) for p, q in zip(xs, ys)):
                return False
        except ValueError:
            return False
    return True


def _limits():
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_S, CPU_S))
    resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_BYTES, FILE_BYTES))
    os.setsid()


def run_tests(code, tests):
    """-> (passed, total, fail reason or None)."""
    tmp = os.path.realpath(tempfile.mkdtemp(prefix="wsp_sbx_"))
    try:
        prof = os.path.join(tmp, "profile.sb")
        root = os.path.dirname(os.path.dirname(os.path.dirname(PY_REAL)))
        open(prof, "w").write(PROFILE.format(tmp=tmp, py=PY, py_real=PY_REAL, py_root=root))
        open(os.path.join(tmp, "main.py"), "w").write(code)
        spent = 0.0
        import time
        for n, t in enumerate(tests):
            if spent > TOTAL_S:
                return n, len(tests), "total time limit"
            inp, out = os.path.join(tmp, "in.txt"), os.path.join(tmp, "out.txt")
            open(inp, "w").write(t["input"])
            t0 = time.time()
            with open(inp) as fi, open(out, "w") as fo:
                p = subprocess.Popen(["/usr/bin/sandbox-exec", "-f", prof, PY, "-I", "-B", "main.py"], cwd=tmp,
                                     stdin=fi, stdout=fo, stderr=subprocess.DEVNULL, preexec_fn=_limits,
                                     env={"PATH": "/usr/bin:/bin", "HOME": tmp, "TMPDIR": tmp})
                try:
                    rc = p.wait(timeout=TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, 9)
                    p.wait()
                    return n, len(tests), "timeout"
            spent += time.time() - t0
            if rc != 0:
                return n, len(tests), f"exit {rc}"
            if not _same(open(out, errors="replace").read(), t["output"]):
                return n, len(tests), "wrong output"
        return len(tests), len(tests), None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def grade_code(text, tests, finish):
    cut = finish == "length"
    code = None if cut else extract_code(text)
    passed, total, fail = (0, len(tests), "no code") if code is None else run_tests(code, tests)
    ok = (not cut) and code is not None and fail is None
    outcome = ("correct" if ok else "cut_at_max_tokens" if cut else "no_answer_finished" if code is None
               else "wrong_answer")
    return {"pred": None if code is None else code[:200], "correct": ok, "outcome": outcome,
            "closed_thinking": "</think>" in text, "loop_end": grade.verbatim_loop_end(text),
            "tests_passed": passed, "tests_total": total, "fail": fail}


def selftest():
    T = [{"input": "3 4\n", "output": "7\n"}, {"input": "10 5\n", "output": "15"}]
    wrap = lambda c: "<think>x</think>\n```python\n" + c + "\n```"  # noqa: E731
    cases = {
        "correct": ("a,b=map(int,input().split());print(a+b)", True),
        "wrong": ("print(0)", False),
        "timeout": ("while True: pass", False),
        "network blocked": ("import socket;socket.create_connection(('1.1.1.1',53),2);print(7)", False),
        "home read blocked": (f"open({os.path.expanduser('~/.zshrc')!r}).read();a,b=map(int,input().split());print(a+b)", False),
        "write outside blocked": ("open('/private/tmp/wsp_escape.txt','w').write('x');a,b=map(int,input().split());print(a+b)", False),
        "exec blocked": ("import subprocess;subprocess.run(['/bin/ls']);a,b=map(int,input().split());print(a+b)", False),
        "temp write allowed": ("open('scratch.txt','w').write('x');a,b=map(int,input().split());print(a+b)", True),
    }
    ok_all = True
    for name, (code, want) in cases.items():
        r = grade_code(wrap(code), T, "stop")
        good = r["correct"] == want
        ok_all &= good
        print(f"{'ok ' if good else 'BAD'} {name:22} correct={r['correct']} fail={r['fail']}")
    print("escape file exists:", os.path.exists("/private/tmp/wsp_escape.txt"))
    print("SELFTEST", "PASSED" if ok_all and not os.path.exists("/private/tmp/wsp_escape.txt") else "FAILED")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
