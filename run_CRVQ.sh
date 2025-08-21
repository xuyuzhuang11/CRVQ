#!/bin/bash
#SBATCH -J CRVQ
#SBATCH -o logs/%j.log
#SBATCH -e logs/%j.err
#SBATCH -p xxx
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH --gres=gpu:1


# conda
. "$HOME"/miniconda3/etc/profile.d/conda.sh
conda activate CRVQ

export CUDA_VISIBLE_DEVICES=0  # or e.g. 0,1,2,3  
# Now supporting 1 GPU temporarily

export MODEL_PATH=/your/path/to/meta-llama/Llama-2-7b


export DATASET_PATH=/your/path/to/CRVQ/data/red_pajama_n=4096_4096_context_length_llama.pth
export SAVE_PATH=/your/path/to/CRVQ/save/Llama-2-7b-8+8+8+8-0.02
export LOG_PATH=$SAVE_PATH


python main.py \
    $MODEL_PATH \
    $DATASET_PATH \
    --nsamples=2048 \
    --val_size=256 \
    --model_seqlen=4096 \
    --num_codebooks=4 \
    --nbits_per_codebook 8 8 8 8 \
    --multibook_ratio=0.02 \
    --shuffle_rule=3 \
    --out_group_size=1 \
    --in_group_size=8 \
    --beam_size=1 \
    --relative_mse_tolerance=0.01 \
    --max_epochs=50 \
    --finetune_lr=3e-5 \
    --finetune_adam_beta1=0.90 \
    --finetune_adam_beta2=0.95 \
    --finetune_keep_best \
    --finetune_batch_size=128 \
    --local_batch_size=4 \
    --finetune_max_epochs=10 \
    --finetune_early_stop=3 \
    --offload_activations \
    --save $SAVE_PATH \
    --resume 

 | tee -a $LOG_PATH/log.txt
