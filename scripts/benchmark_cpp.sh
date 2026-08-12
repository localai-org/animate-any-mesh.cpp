#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 8 ]]; then
  echo "usage: $0 AAM_CLI MODEL.gguf INPUT.glb PROMPT [RUNS=5] [DEVICE=vulkan] [MAX_VERTICES=4096] [TRAJECTORIES=auto]" >&2
  exit 2
fi

cli="$1"
model="$2"
input="$3"
prompt="$4"
runs="${5:-5}"
device="${6:-vulkan}"
max_vertices="${7:-4096}"
trajectories="${8:-0}"

if [[ ! "$runs" =~ ^[1-9][0-9]*$ || ! "$max_vertices" =~ ^[1-9][0-9]*$ || ! "$trajectories" =~ ^[0-9]+$ ]]; then
  echo "RUNS and MAX_VERTICES must be positive integers; TRAJECTORIES must be a non-negative integer" >&2
  exit 2
fi

work_dir="$(mktemp -d -t aam-benchmark.XXXXXXXX)"
trap 'rm -rf -- "$work_dir"' EXIT

run_once() {
  local label="$1"
  local output="$work_dir/$label.glb"
  local start end
  local -a trajectory_args=()
  if [[ "$trajectories" != 0 ]]; then
    trajectory_args=(--trajectories "$trajectories")
  fi
  start="$(date +%s%N)"
  "$cli" animate --model "$model" --input "$input" --output "$output" \
    --prompt "$prompt" --device "$device" --seed 666 --steps 64 --guidance 3.0 \
    --max-vertices "$max_vertices" "${trajectory_args[@]}" \
    >/dev/null
  end="$(date +%s%N)"
  awk -v start="$start" -v end="$end" 'BEGIN { printf "%.6f\n", (end-start)/1000000000 }'
}

echo "warming up $device" >&2
run_once warmup >/dev/null
times=()
for ((run = 1; run <= runs; ++run)); do
  elapsed="$(run_once "run-$run")"
  times+=("$elapsed")
  printf 'run %d seconds %s\n' "$run" "$elapsed"
done
printf '%s\n' "${times[@]}" | sort -n | awk '
  { samples[NR] = $1 }
  END {
    if (NR % 2) median = samples[(NR + 1) / 2]
    else median = (samples[NR / 2] + samples[NR / 2 + 1]) / 2
    printf "median seconds %.6f\n", median
  }'
