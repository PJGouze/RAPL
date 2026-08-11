#!/usr/bin/env bash
#SBATCH --job-name=rapl-toy-pilot
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=/mnt/beegfs/home/pjgouze/RAPL-env/RAPL/pilot_runs/slurm-toy-pilot-%j.out
#SBATCH --error=/mnt/beegfs/home/pjgouze/RAPL-env/RAPL/pilot_runs/slurm-toy-pilot-%j.err

# REQUIRED SITE VALUES — replace these placeholders before submission because
# no verified partition, account, CPU, or memory settings exist in this repo:
#SBATCH --partition=__SET_CLUSTER_PARTITION__
#SBATCH --account=__SET_CLUSTER_ACCOUNT__
#SBATCH --cpus-per-task=__SET_CPUS_PER_TASK__
#SBATCH --mem=__SET_MEMORY__

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

readonly WORKDIR="/mnt/beegfs/home/pjgouze/RAPL-env/RAPL"
readonly PYTHON_EXECUTABLE="${RAPL_PYTHON_EXECUTABLE:?Export RAPL_PYTHON_EXECUTABLE as the verified cluster Python executable}"
readonly PILOT_ROOT="${WORKDIR}/pilot_runs/toy_gpu_seed17_${SLURM_JOB_ID}"
readonly TRAIN_OUTPUT="${PILOT_ROOT}/training"

test -x "${PYTHON_EXECUTABLE}"
test -d "${WORKDIR}/data_files/toy/processed"
test ! -e "${PILOT_ROOT}"
mkdir -p "${PILOT_ROOT}"
cd "${WORKDIR}"

find data_files/toy -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum > "${PILOT_ROOT}/toy_data_before.sha256"

"${PYTHON_EXECUTABLE}" main.py \
    --dataset_name toy \
    --device 0 \
    --epochs 5 \
    --batch_size 2 \
    --num_workers 0 \
    --seed 17 \
    --output_dir "${TRAIN_OUTPUT}"

"${PYTHON_EXECUTABLE}" -m scripts.validate_toy_pilot \
    --output-dir "${TRAIN_OUTPUT}" \
    --expected-epochs 5

sha256sum -c "${PILOT_ROOT}/toy_data_before.sha256" \
    > "${PILOT_ROOT}/toy_data_hash_check_after.txt"
