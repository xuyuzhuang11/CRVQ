#!/bin/bash
#SBATCH -J CRVQ_lmeval
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



# for 0-shot evals

python lmeval.py \
    --model hf \
    --model_args pretrained=$MODEL_PATH,dtype=float16,parallelize=True \
    --tasks winogrande,piqa,hellaswag,arc_easy,arc_challenge,boolq \
    --batch_size auto \
    --aqlm_checkpoint_path $QUANTIZED_MODEL # if evaluating quantized model

