#!/bin/bash
#
#SBATCH --job-name=bert-cls
#SBATCH --output=logscls/bert_class_%j.txt  # output file
#SBATCH -e logscls/bert_class_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=2080ti-long # Partition to submit to
#SBATCH --mem=40GB
#
#SBATCH --ntasks=1

export DATASET=w2
export RANDOM_SEQ_LEN=3
export BERT_MODEL='/mnt/nfs/work1/696ds-s20/abajaj/nlplab/long-term-context/models/bert-large-cased'

export TRAIN=./data/${DATASET}-classifier-data${RANDOM_SEQ_LEN}-train.csv
export TEST=./data/${DATASET}-classifier-data${RANDOM_SEQ_LEN}-test.csv

export OUT_DIR=./${DATASET}/${RANDOM_SEQ_LEN}

python -u bert_finetuning.py --trainFile=${TRAIN} --devFile=${TEST} --testFile=${TEST} --outputDir=${OUT_DIR} --model_name_or_path ${BERT_MODEL} --gpu