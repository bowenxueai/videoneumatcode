#!/bin/bash
# Usage: ./launch_render_btf.sh <TOTAL_MATERIALS> [ARRAY_SPEC] [RESOLUTION] [TRAJ]
# Example: ./launch_render_btf.sh 5700 0-99 1024 sparse

set -e
cd "$(dirname "$0")/../.."

TOTAL_MATERIALS="${1:?Usage: $0 <TOTAL_MATERIALS> [ARRAY_SPEC] [RESOLUTION] [TRAJ]}"
ARRAY_SPEC="${2:-0-7}"
RESOLUTION="${3:-1024}"
TRAJ="${4:-sparse}"

JOB_ID=$(sbatch -A nvr_rtr_difftracer --array="$ARRAY_SPEC" --export=ALL,TOTAL_MATERIALS=$TOTAL_MATERIALS,RESOLUTION=$RESOLUTION,TRAJ=$TRAJ --parsable sample_scripts/cluster/render_btf.sh)
echo "Render job: $JOB_ID"

UNIFY_ID=$(sbatch --dependency=afterok:$JOB_ID --job-name="unify_$JOB_ID" --parsable sample_scripts/cluster/unify_btf.sh "$JOB_ID")
echo "Unify job: $UNIFY_ID (runs after $JOB_ID)"
echo "Output: outputs/slurm/render_btf/$JOB_ID/"
