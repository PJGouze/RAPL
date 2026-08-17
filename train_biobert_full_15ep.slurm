#!/bin/bash
#SBATCH --job-name=biobert_full_15ep
#SBATCH --partition=all
#SBATCH --nodelist=n101
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_biobert_full_%j.out
#SBATCH --error=slurm_biobert_full_%j.err

set -euo pipefail

echo "========================================"
echo "BioBERT full training"
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : $(hostname)"
echo "Start     : $(date)"
echo "========================================"

cd /mnt/beegfs/home/pjgouze/RAPL-env/RAPL

# Activate conda
source /mnt/beegfs/home/pjgouze/miniconda3/etc/profile.d/conda.sh
conda activate rapl-env

echo
echo "Python:"
which python
python --version

echo
echo "GPU:"
nvidia-smi

echo
echo "Git HEAD:"
git rev-parse HEAD

echo
echo "Checking frozen critical code..."
sha256sum -c pre_biobert_full_experiment_code.sha256

OUTPUT_DIR="experiments/biobert_full_15ep_seed17"

if [ -e "$OUTPUT_DIR" ]; then
    echo "ERROR: output directory already exists: $OUTPUT_DIR"
    exit 1
fi

echo
echo "========================================"
echo "Starting training"
echo "Output: $OUTPUT_DIR"
echo "========================================"

time python main.py \
    --model_type GCN \
    --num_layers 2 \
    --emb_size 768 \
    --hidden_dims 512 \
    --out_dims 512 \
    --dropout 0.2 \
    --lr 1e-3 \
    --epochs 15 \
    --device 0 \
    --dataset_name OntoOmicsKG_step2 \
    --batch_size 4 \
    --num_workers 0 \
    --seed 17 \
    --final-dataset-dir \
        data_files/OntoOmicsKG_step2/final_biobert_full_v2 \
    --output_dir \
        "$OUTPUT_DIR" \
    --pathTrainAfterEpoch 0

echo
echo "========================================"
echo "Training finished successfully"
echo "End: $(date)"
echo "========================================"
