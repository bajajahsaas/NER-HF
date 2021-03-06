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

export DATASET=public # w2, public
export RANDOM_SEQ_LEN=200 # 5, 200
export BERT_MODEL='/mnt/nfs/work1/696ds-s20/abajaj/nlplab/long-term-context/models/bert-large-cased'

export MODE=no-address # all, no-address
export NUM_LABELS=3 # 4, 3

export TRAIN=./data/${MODE}/${DATASET}-classifier-data-train${RANDOM_SEQ_LEN}.csv
export TEST=./data/${MODE}/${DATASET}-classifier-data-test${RANDOM_SEQ_LEN}.csv

export OUT_DIR=./${DATASET}/${MODE}/${RANDOM_SEQ_LEN}

python -u bert_finetuning.py --trainFile=${TRAIN} --devFile=${TEST} --testFile=${TEST} --outputDir=${OUT_DIR} --model_name_or_path ${BERT_MODEL} --num_labels=${NUM_LABELS} --gpu
