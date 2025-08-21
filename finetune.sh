#!/bin/bash
#SBATCH -J finetune
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




export CUDA_VISIBLE_DEVICES=0

export MODEL_PATH=/your/path/to/meta-llama/Llama-2-7b


export DATASET_PATH=/your/path/to/CRVQ/data/red_pajama_n=4096_4096_context_length_llama.pth
export INPUT_PATH=/your/path/to/CRVQ/save/Llama-2-7b-8+8+8+8-0.02
export SAVE_PATH=/your/path/to/CRVQ/save/finetune/Llama-2-7b-8+8+8+8-0.02

python finetune.py \
  --base_model $MODEL_PATH \
  --quant_model $INPUT_PATH \
  --dataset $DATASET_PATH \
  --model_seqlen=4096 \
  --eval_model_seqlen=4096 \
  --eval_datasets wikitext2 \
  --nsamples=1024 \
  --val_size=64 \
  --lr=1e-5 \
  --adam_beta1=0.90 \
  --adam_beta2=0.999 \
  --epochs=2 \
  --early_stop=3 \
  --batch_size=16 \
  --microbatch_size=1 \
  --save $SAVE_PATH \
  --gradient_checkpointing \
  --amp \
  --device_map auto