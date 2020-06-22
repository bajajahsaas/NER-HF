#!/bin/bash
#
#SBATCH --job-name=bert-cls
#SBATCH --output=logscls/bert_class_%j.txt  # output file
#SBATCH -e logscls/bert_class_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=2080ti-short # Partition to submit to
#SBATCH --mem=40GB
#
#SBATCH --ntasks=1
export DATASET=arithmetic
export BERT_MODEL='/mnt/nfs/work1/696ds-s20/abajaj/nlplab/long-term-context/models/bert-large-cased'
export NUM_LABELS=2

export TRAIN=./data/${DATASET}-train.csv
export TEST=./data/${DATASET}-test.csv

export OUT_DIR=./${DATASET}

python -u bert_finetuning.py --trainFile=${TRAIN} --devFile=${TEST} --testFile=${TEST} --outputDir=${OUT_DIR} --model_name_or_path ${BERT_MODEL} --num_labels=${NUM_LABELS} --gpu
