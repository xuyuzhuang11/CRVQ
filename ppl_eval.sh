#!/bin/bash
#SBATCH -J CRVQ_ppl
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

export MODEL_PATH=/your/path/to/meta-llama/Llama-2-7b
export QUANTIZED_MODEL=/your/path/to/CRVQ/save/Llama-2-7b-8+8+8+8-0.02

python eval.py \
  --model_path $MODEL_PATH \
  --quant_path $QUANTIZED_MODEL \
  --seqlen 4096