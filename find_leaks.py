import glob
import subprocess
import time


def main():
    test_files = glob.glob("tests/**/*.py", recursive=True)
    leaky_files = []

    for f in test_files:
        if not f.endswith(".py") or "__pycache__" in f or "test_graph.py" in f:
            continue
        print(f"Testing {f}...")
        start = time.time()
        try:
            res = subprocess.run(
                ["uv", "run", "pytest", f, "-m", "not live", "-q", "--timeout=10"],
                capture_output=True,
                timeout=15,
            )
            dur = time.time() - start
            if res.returncode != 0:
                out = res.stdout.decode()
                if "no tests collected" in out:
                    print(f"  -> OK (No tests, {dur:.1f}s)")
                elif "Failed: Timeout" in out or "pytest-timeout" in out:
                    print(f"  -> HUNG (pytest-timeout, {dur:.1f}s)")
                    leaky_files.append(f)
                else:
                    print(f"  -> Failed (exit {res.returncode}, {dur:.1f}s)")
            else:
                print(f"  -> OK ({dur:.1f}s)")
        except subprocess.TimeoutExpired:
            print("  -> HUNG (subprocess timeout)")
            leaky_files.append(f)

    print("\nLeaky files found:")
    for f in leaky_files:
        print(f)


if __name__ == "__main__":
    main()
