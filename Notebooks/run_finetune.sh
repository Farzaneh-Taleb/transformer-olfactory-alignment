#!/usr/bin/env bash
#SBATCH --output=/Midgard/home/%u/phd/Olfaction/MoLFormer_N2024/logs/%J_slurm.out
#SBATCH --error=/Midgard/home/%u/phd/Olfaction/MoLFormer_N2024/logs/%J_slurm.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fatn@kth.se
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --constrain="eowyn|arwen"

#SBATCH --array=1-31%10

ct_array=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)



# Check job environment
echo "JOB:  ${SLURM_JOB_ID}"
echo "TASK: ${SLURM_ARRAY_TASK_ID}"
echo "HOST: $(hostname)"
echo ""

# Activate conda
source "${HOME}/mambaforge/etc/profile.d/conda.sh"
conda activate Mol

## Train
#SOURCE_PATH="${HOME}/rpl-workshop"
#RUNS_PATH="${HOME}/rpl-workshop/runs"
#DATA_PATH="/local_storage/datasets/CUB_20"

ct=${ct_array[`expr $SLURM_ARRAY_TASK_ID % ${#ct_array[@]}`]}
python finetune_multitask.py \
        --device cuda \
        --batch_size 32  \
        --n_head 12 \
        --n_layer 12 \
        --n_embd 768 \
        --d_dropout 0.1 \
        --dropout 0.1 \
        --lr_start 3e-5 \
        --num_workers 8\
        --max_epochs 50 \
        --num_feats 32 \
        --seed_path '../MoLformer_Pretrained/checkpoints/N-Step-Checkpoint_3_30000.ckpt' \
        --dataset_name curated_GS_LF_merged_4983 \
        --dataset_index $ct \
        --base_path /local_storage/datasets/farzaneh/alignment_olfaction_datasets/curated_datasets \
        --data_root /mols_datasets \
        --input_file_indices '/embeddings/pom/gslf-splits-idx-' \
        --input_file_data '/embeddings/pom/gslf_pom_proba_' \
        --checkpoints_folder './checkpoints_gs_lf'\
        --dims 768 768 768 1
