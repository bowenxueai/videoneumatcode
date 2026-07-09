#!/bin/bash
#SBATCH --nodes=1
#SBATCH -A nvr_rtr_rgbxnext
#SBATCH --partition=cpu_short
#SBATCH -t 00:30:00
#SBATCH --output=outputs/slurm/render_btf/%x_unify.log

# Usage: ./unify_btf.sh <JOB_ID>

JOB_ID="${1:?Usage: $0 <JOB_ID>}"
BASE="/lustre/fsw/portfolios/nvr/users/shadadan/neugen/outputs/slurm/render_btf/${JOB_ID}"
OUT="$BASE/all_materials"

mkdir -p "$OUT"

# Move all material dirs from task folders
for d in "$BASE"/*/; do
    [[ "$(basename "$d")" =~ ^[0-9]+$ ]] || continue
    mv "$d"/*/ "$OUT/" 2>/dev/null
done

# Merge all CSVs of each type
for csv_name in metadata_clean.csv displacement_stats1.csv; do
    first=true
    for d in "$BASE"/*/; do
        [[ "$(basename "$d")" =~ ^[0-9]+$ ]] || continue
        csv="$d/$csv_name"
        [ -f "$csv" ] || continue
        if $first; then
            cat "$csv"
            first=false
        else
            tail -n +2 "$csv"
        fi
    done > "$OUT/$csv_name"
done

# Copy first material's scene_state as 0_scene_state
first_scene=$(find "$OUT" -maxdepth 2 -type d -name "scene_state" | head -1)
[ -n "$first_scene" ] && cp -r "$first_scene" "$OUT/0_scene_state"

echo "Done: $(ls -d "$OUT"/*/ 2>/dev/null | wc -l) materials in $OUT"
[ -d "$OUT/0_scene_state" ] && echo "Scene state copied to: $OUT/0_scene_state"
