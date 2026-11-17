"""Run every Arborist test and print pass/fail with the time each took.

    python tests/run_all.py            (GUI tests need a display; on Linux: xvfb-run -a python tests/run_all.py)
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["make_synthetic", "test_core", "test_classify", "test_ais", "test_dome", "test_thin_stack",
         "test_messy_strokes", "test_loft_artifacts", "test_storage", "test_user_sheet", "test_review", "test_features", "test_autotrace",
         "test_gui", "test_database", "test_database_usb", "test_pen_real_mouse", "test_ephys", "test_pool"]


def main():
    env = dict(os.environ, MPLBACKEND="Agg")
    only = sys.argv[1:]
    failed = []
    for name in ORDER:
        path = os.path.join(HERE, name + ".py")
        if not os.path.exists(path) or (only and name not in only and name != "make_synthetic"):
            continue
        t = time.perf_counter()
        try:
            r = subprocess.run([sys.executable, "-W", "ignore", path], cwd=HERE, env=env, capture_output=True,
                               text=True, timeout=900)
            ok, out = r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()
        except subprocess.TimeoutExpired:
            ok, out = False, ["timed out"]
        dt = time.perf_counter() - t
        last = out[-1] if out else ""
        print(f"{'ok  ' if ok else 'FAIL'} {name:22s} {dt:6.1f}s  {last[:90]}", flush=True)
        if not ok:
            failed.append(name)
            print("     " + "\n     ".join(out[-8:]), flush=True)
    print("\nALL TESTS PASSED" if not failed else f"\nFAILED: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
