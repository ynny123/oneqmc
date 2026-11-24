#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 || $# > 3 )) || [[ $# == 3 && $3 != --smoke ]]; then
    echo 'Usage: bash run.sh {ch4|c2h6} OUTPUT_DIR [--smoke]' >&2
    exit 2
fi

case "$1" in
    ch4) geometry=CH4/CH4.yaml ;;
    c2h6) geometry=C2H6_bbmep_00/00.yaml ;;
    *) echo "Unknown example: $1" >&2; exit 2 ;;
esac

steps=50000
eval_steps=1000
checkpoint_interval=10000
sampler_args=()
if [[ ${3:-} == --smoke ]]; then
    steps=2
    eval_steps=2
    checkpoint_interval=2
    sampler_args=(--max-eq-steps 1)
fi

output=$(realpath -m -- "$2")
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd -- "$repo"
export AMLT_DATA_DIR="$repo/data"
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
data_file="$AMLT_DATA_DIR/$geometry"
reference=$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["extras"]["reference_energy"])' "$data_file")

mkdir -p -- "$(dirname -- "$output")"
mkdir -- "$output"
train_dir="$output/fit"
eval_dir="$output/evaluation"
common=(-d "${geometry%/*}" -a orbformer-se --mol-batch-size 1 --electron-batch-size 1024
    --pretrain-steps 0 --no-autoresume --max-restarts 1 --metric-logger-period 1)
curves=(python -u examples/plot_absolute_energy_error.py --run-dir "$train_dir" --reference-energy="$reference")

"${curves[@]}" --watch >"$output/curves_live.log" 2>&1 &
watch_pid=$!
stop_watch() {
    kill "$watch_pid" 2>/dev/null || true
    wait "$watch_pid" 2>/dev/null || true
}
trap stop_watch EXIT

SECONDS=0
python -u scripts/transferable.py "${common[@]}" "${sampler_args[@]}" --seed 42 \
    -n "$steps" --chkpts-slow-interval "$checkpoint_interval" -w "$train_dir"
printf '%s\n' "$SECONDS" >"$output/training_wall_seconds.txt"
stop_watch
trap - EXIT

checkpoint="$train_dir/training/chkpt-$steps.pt"
test -s "$checkpoint"

SECONDS=0
ORBFORMER_PICKLE_LOADING=1 python -u scripts/transferable.py "${common[@]}" "${sampler_args[@]}" \
    --test --discard-sampler-state --seed 20261119 -c "$checkpoint" -n "$eval_steps" \
    --chkpts-fast-interval "$eval_steps" --chkpts-slow-interval "$eval_steps" -w "$eval_dir"
printf '%s\n' "$SECONDS" >"$output/evaluation_wall_seconds.txt"
python examples/single_geometry_hydrocarbons/summarize_evaluation.py "$eval_dir" "$data_file"

"${curves[@]}"
test "$(wc -l <"$train_dir/analysis/absolute_energy_error_online.csv")" -eq "$((steps + 1))"
