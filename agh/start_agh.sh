#!/usr/bin/env bash
set -euo pipefail

VASC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGH_ROOT="${1:-$(cd "${VASC_ROOT}/.." && pwd)/agnes-harness}"
PYTHON_BIN="${VASC_PYTHON:-$(command -v python || true)}"

if [[ ! -f "${AGH_ROOT}/packages/cli/dist/local/agnes.mjs" ]]; then
  echo "AGH local build not found: ${AGH_ROOT}/packages/cli/dist/local/agnes.mjs" >&2
  echo "Build Agnes Harness first by following docs/HACKATHON_AGH_GUIDE.md." >&2
  exit 2
fi

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found. Activate the x_gaussian environment first." >&2
  exit 2
fi

export VASC_GAUSSIAN_ROOT="${VASC_ROOT}"
export VASC_PYTHON="${PYTHON_BIN}"

echo "Vasc-Gaussian root: ${VASC_GAUSSIAN_ROOT}"
echo "Python: ${VASC_PYTHON}"
echo "AGH root: ${AGH_ROOT}"
echo "The Agnes model key must be entered in Web settings; this script never reads it."

cd "${VASC_ROOT}"
exec node "${AGH_ROOT}/packages/cli/dist/local/agnes.mjs" serve
