#!/usr/bin/env bash
set -u

# Read-only environment report for the VascPilot / Agnes Harness deployment.
# It intentionally does not print environment variable values, tokens, or keys.

repo_root="${VASC_GAUSSIAN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
agh_root="${AGNES_HARNESS_ROOT:-/root/autodl-tmp/agnes-harness}"
python_bin="${VASC_PYTHON:-python}"

section() {
  printf '\n[%s]\n' "$1"
}

command_report() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then
    printf '%-18s OK  %s\n' "$name" "$(command -v "$name")"
  else
    printf '%-18s MISSING\n' "$name"
  fi
}

printf 'VascPilot server doctor (read-only)\n'
printf 'Generated: %s\n' "$(date -Iseconds 2>/dev/null || date)"

section "System"
uname -a 2>/dev/null || true
df -h "$repo_root" 2>/dev/null | tail -n +1 || true

section "Commands"
for cmd in git node npm pnpm conda nvidia-smi; do
  command_report "$cmd"
done
if command -v node >/dev/null 2>&1; then node --version 2>/dev/null || true; fi
if command -v pnpm >/dev/null 2>&1; then pnpm --version 2>/dev/null || true; fi

section "Python"
printf 'requested binary   %s\n' "$python_bin"
if "$python_bin" --version >/dev/null 2>&1; then
  "$python_bin" --version 2>&1
  "$python_bin" - <<'PY'
import importlib.util
import sys

print("executable         ", sys.executable)
for package in ("torch", "yaml", "numpy", "skimage", "tqdm"):
    print(f"{package:<18}", "OK" if importlib.util.find_spec(package) else "MISSING")
try:
    import torch
    print("torch version      ", torch.__version__)
    print("cuda available    ", torch.cuda.is_available())
    print("cuda device count ", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("cuda device 0     ", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch inspection  ERROR", type(exc).__name__, str(exc)[:200])
PY
else
  printf 'Python binary is not runnable.\n'
fi

section "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>&1 || true
else
  printf 'nvidia-smi is unavailable.\n'
fi

section "Vasc-Gaussian repository"
printf 'repo root          %s\n' "$repo_root"
if git -C "$repo_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  printf 'branch             %s\n' "$(git -C "$repo_root" branch --show-current 2>/dev/null)"
  printf 'commit             %s\n' "$(git -C "$repo_root" rev-parse --short HEAD 2>/dev/null)"
  printf 'working tree       '
  if [ -z "$(git -C "$repo_root" status --porcelain 2>/dev/null)" ]; then
    printf 'clean\n'
  else
    printf 'has local changes\n'
  fi
else
  printf 'NOT A GIT REPOSITORY\n'
fi

for relative in \
  "train.py" \
  "agh/bridge.py" \
  "agh/profiles/aneurism-v4.json" \
  "agh/vasc-tools/index.mjs" \
  ".agh/skills/vasc-reconstruction/SKILL.md" \
  "configs/aneurism_lowres_v4_scalemax_loose.yaml"; do
  if [ -e "$repo_root/$relative" ]; then
    printf '%-50s OK\n' "$relative"
  else
    printf '%-50s MISSING\n' "$relative"
  fi
done

section "Agnes Harness"
printf 'AGH root           %s\n' "$agh_root"
if [ -f "$agh_root/packages/cli/dist/local/agnes.mjs" ]; then
  printf 'local CLI build    OK\n'
elif [ -f "$agh_root/package.json" ]; then
  printf 'source checkout    OK, CLI build missing\n'
else
  printf 'checkout           MISSING\n'
fi

section "Data hints"
for candidate in \
  "$repo_root/data" \
  "$repo_root/dataset" \
  "/root/autodl-tmp/data"; do
  if [ -e "$candidate" ]; then
    printf '%-50s PRESENT\n' "$candidate"
  else
    printf '%-50s absent\n' "$candidate"
  fi
done

section "Secret safety"
if [ -n "${AGNES_MODEL_KEY:-}" ] || [ -n "${OPENAI_API_KEY:-}" ]; then
  printf 'A model-key environment variable is set (value hidden).\n'
else
  printf 'No common model-key environment variable is visible in this shell.\n'
fi
printf 'Do not paste any key value into chat, screenshots, Git, logs, or demo video.\n'

section "Next"
printf 'Send this report to the setup assistant. It contains no secret values by design.\n'
