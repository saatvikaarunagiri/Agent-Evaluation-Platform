"""`make doctor` — check your machine has what the monorepo needs.
Prints a green/red checklist and never crashes.
"""
from __future__ import annotations
import shutil, subprocess, sys, platform, urllib.request

OK, BAD, WARN = "\033[92m✓\033[0m", "\033[91m✗\033[0m", "\033[93m!\033[0m"


def cmd(name: str, args: list[str]) -> bool:
    if not shutil.which(name):
        print(f"  {BAD} {name:<14} not found on PATH")
        return False
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=15)
        v = (out.stdout or out.stderr).strip().splitlines()[0][:60]
        print(f"  {OK} {name:<14} {v}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  {WARN} {name:<14} present but errored: {e}")
        return True


def port(host: str, p: int, label: str) -> None:
    import socket
    s = socket.socket(); s.settimeout(1.0)
    try:
        s.connect((host, p)); print(f"  {OK} {label:<14} reachable on {host}:{p}")
    except Exception:
        print(f"  {WARN} {label:<14} not reachable on {host}:{p} (start its docker profile)")
    finally:
        s.close()


def main() -> int:
    print(f"\nMachine: {platform.platform()}  |  Python {sys.version.split()[0]}  "
          f"|  arch {platform.machine()}\n")

    print("Required tooling")
    have_py = sys.version_info >= (3, 11)
    print(f"  {OK if have_py else BAD} python3        {sys.version.split()[0]} "
          f"({'>=3.11 ok' if have_py else 'need >=3.11'})")
    cmd("docker", ["docker", "--version"])
    cmd("node", ["node", "--version"])
    cmd("npm", ["npm", "--version"])

    print("\nOptional (free local model backend)")
    if shutil.which("ollama"):
        cmd("ollama", ["ollama", "--version"])
    else:
        print(f"  {WARN} ollama         not installed (fine — only needed for local LLM/judge)")

    print("\nApple-Silicon notes")
    if platform.machine() == "arm64" and sys.platform == "darwin":
        print(f"  {OK} Apple Silicon detected — MLX/ONNX CoreML paths available")
    else:
        print(f"  {WARN} not on Apple Silicon — that's fine, CPU paths used instead")

    print("\nInfra ports (only the ones for the profile you started will be up)")
    port("localhost", 9092, "redpanda")
    port("localhost", 8123, "clickhouse")

    print("\nDone. Red ✗ = must fix. Yellow ! = only needed for some features.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
