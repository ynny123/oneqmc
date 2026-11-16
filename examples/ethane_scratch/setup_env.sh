#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
python_bin="${PYTHON_BIN:-python3.11}"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
if [[ ! -x "$ONEQMC_VENV/bin/python" ]]; then
  "$python_bin" -m venv "$ONEQMC_VENV"
fi
pip_source=(--index-url "${ONEQMC_PIP_INDEX_URL:-https://pypi.org/simple}")
if [[ -n "${ONEQMC_WHEELHOUSE:-}" ]]; then
  pip_source=(--no-index --find-links "$ONEQMC_WHEELHOUSE")
fi
"$ONEQMC_VENV/bin/python" -m pip install "${pip_source[@]}" --progress-bar off \
  -r "$ONEQMC_REPO_ROOT/requirements/ethane-a100-py311.txt" \
  --report "$ONEQMC_WORK_ROOT/install-report.json"
"$ONEQMC_VENV/bin/python" -m pip check
printf 'Environment ready. Activate with:\nsource %q/bin/activate\n' "$ONEQMC_VENV"
