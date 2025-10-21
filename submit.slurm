#!/bin/bash

#SBATCH --time=5:00:00
#SBATCH --job-name=prepare
#SBATCH -o ./temp/%x.out
#SBATCH --account=your_account_name
#SBATCH -q inferno
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH --mem-per-gpu=80G

args=("$@")

sanitize_component() {
    local raw=$1
    # Keep alphanumeric, dot, dash, and underscore; collapse everything else to underscores.
    echo "$raw" | tr -cs '[:alnum:]._-' '_' | sed 's/^_//; s/_$//'
}

job_name=""
for arg in "${args[@]}"; do
    trimmed="${arg#--}"
    trimmed="${trimmed#-}"
    component=$(sanitize_component "$trimmed")
    if [[ -n "$component" ]]; then
        if [[ -z "$job_name" ]]; then
            job_name="$component"
        else
            job_name="${job_name}-${component}"
        fi
    fi
done

if [[ -z "$job_name" ]]; then
    job_name="prepare"
fi

if [[ -n "$SLURM_JOB_ID" ]]; then
    if command -v scontrol >/dev/null 2>&1; then
        scontrol update JobId="$SLURM_JOB_ID" JobName="$job_name" >/dev/null 2>&1 || true
    fi
    echo "Using job name: ${job_name}"
fi

module load anaconda3
module load cuda/12.6.1
conda activate lm
ulimit -c 0  # disable core dumps
python main_prep.py "$@"
#python main_imitexp.py "$@"
#python main_predexp.py "$@"
