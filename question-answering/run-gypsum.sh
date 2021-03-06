#!/bin/bash
#
#SBATCH --job-name=qa
#SBATCH --output=logsqa/qa.txt  # output file
#SBATCH -e logsqa/qa.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=2080ti-long # Partition to submit to
#SBATCH --mem=40GB
#
#SBATCH --ntasks=1

export SQUAD_DIR='/mnt/nfs/work1/696ds-s20/abajaj/instabase/data/SQUAD1.0'
export BERT_MODEL='/mnt/nfs/work1/696ds-s20/abajaj/nlplab/long-term-context/models/bert-base-uncased'
export OUTPUT_DIR='/mnt/nfs/work1/696ds-s20/abajaj/instabase/models/qa-squad1.0'

python run_squad.py \
  --model_type bert \
  --model_name_or_path $BERT_MODEL \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $OUTPUT_DIR