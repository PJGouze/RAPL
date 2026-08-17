#!/bin/bash

#SBATCH --job-name=rapl_ontoomics
#SBATCH --gres=gpu:1
#SBATCH --mem=300G
#SBATCH --time=23:00:00
#SBATCH --output=slurm_ontoomics_%j.out
#SBATCH --error=slurm_ontoomics_%j.err

set -e

cd /mnt/beegfs/home/pjgouze/RAPL-env/RAPL

source /mnt/beegfs/home/pjgouze/miniconda3/etc/profile.d/conda.sh
conda activate rapl-env

echo "===== RAPL TRAINING ====="
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "Start time : $(date)"
echo "Python     : $(which python)"
echo "========================="

python main.py \
    --dataset_name OntoOmicsKG_step2 \
    --model_type GCN \
    --bidirectional \
    --use_stop_mlp \
    --num_layers 2 \
    --num_heads 4 \
    --K 3 \
    --in_dims 1152 \
    --emb_size 384 \
    --hidden_dims 512 \
    --out_dims 512 \
    --dropout 0.2 \
    --lr 0.001 \
    --epochs 15 \
    --batch_size 4 \
    --num_workers 0 \
    --device 0 \
    --pathTrainAfterEpoch 0 \
    --wandb_id full_run_fp32 \
    --output_dir experiments/OntoOmicsKG_step2/full_run_fp32

echo "========================="
echo "Training finished."
echo "End time : $(date)"
echo "========================="
