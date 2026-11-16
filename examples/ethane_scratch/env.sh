#!/usr/bin/env bash
# Source this file from any directory. All writable state stays on project storage.
export ONEQMC_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export ONEQMC_WORK_ROOT="${ONEQMC_WORK_ROOT:-$ONEQMC_REPO_ROOT/.runtime/ethane}"
export ONEQMC_WORK_ROOT="$(realpath -m -- "$ONEQMC_WORK_ROOT")"
case "$ONEQMC_WORK_ROOT" in
  /root|/root/*)
    echo 'Please set ONEQMC_WORK_ROOT to a writable project/data disk outside /root.' >&2
    return 1 2>/dev/null || exit 1
    ;;
esac
export ONEQMC_VENV="${ONEQMC_VENV:-$ONEQMC_WORK_ROOT/venv}"
export ONEQMC_VENV="$(realpath -m -- "$ONEQMC_VENV")"
case "$ONEQMC_VENV" in
  /root|/root/*)
    echo 'Please set ONEQMC_VENV to a project/data disk outside /root.' >&2
    return 1 2>/dev/null || exit 1
    ;;
esac
export ONEQMC_CACHE_ROOT="$ONEQMC_WORK_ROOT/cache"
export TMPDIR="$ONEQMC_CACHE_ROOT/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export XDG_CACHE_HOME="$ONEQMC_CACHE_ROOT/xdg"
export XDG_CONFIG_HOME="$ONEQMC_CACHE_ROOT/config"
export XDG_DATA_HOME="$ONEQMC_CACHE_ROOT/data"
export PIP_CACHE_DIR="$ONEQMC_CACHE_ROOT/pip"
export PIP_CONFIG_FILE=/dev/null
export PIP_DISABLE_PIP_VERSION_CHECK=1
export CONDA_PKGS_DIRS="$ONEQMC_CACHE_ROOT/conda"
export CONDA_ENVS_PATH="$ONEQMC_WORK_ROOT/conda-envs"
export CONDA_REGISTER_ENVS=false
export PYTHONPYCACHEPREFIX="$ONEQMC_CACHE_ROOT/pycache"
export CUDA_CACHE_PATH="$ONEQMC_CACHE_ROOT/cuda"
export TRITON_CACHE_DIR="$ONEQMC_CACHE_ROOT/triton"
export NUMBA_CACHE_DIR="$ONEQMC_CACHE_ROOT/numba"
export JAX_COMPILATION_CACHE_DIR="$ONEQMC_CACHE_ROOT/jax"
export MPLCONFIGDIR="$ONEQMC_CACHE_ROOT/matplotlib"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
# Enables reading this experiment's own checkpoints for resume and evaluation.
export ORBFORMER_PICKLE_LOADING=1
export PYTHONPATH="$ONEQMC_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" \
  "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$PYTHONPYCACHEPREFIX" \
  "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$NUMBA_CACHE_DIR" \
  "$JAX_COMPILATION_CACHE_DIR" "$MPLCONFIGDIR"
