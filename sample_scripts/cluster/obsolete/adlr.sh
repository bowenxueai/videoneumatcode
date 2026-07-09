PROJ=$DISK/neugen
init="source /opt/miniconda/bin/activate neugen"
SLURM_JOB_ID=haha2node
script_name=$1
command="$init && cd $PROJ && bash scripts/$script_name.sh out_dir='\${exp_name}' out_root='outputs/slurm/$SLURM_JOB_ID/\${exp_name}'"


submit_job --account nvr_rtr_rgbxnext \
    --nodes 2 \
    --gpu 8 \
    --cpu 24 \
    --mem 128 \
    --partition batch \
    --duration 4 \
    --mounts $DISK/neugen:/home/NeuGen,/lustre:/lustre \
    -n lrm_jobs \
    --image /lustre/fsw/portfolios/nvr/users/shadadan/cache/neugen.sqsh \
    --no_cache_sqsh \
    -c "bash -c '$command'"