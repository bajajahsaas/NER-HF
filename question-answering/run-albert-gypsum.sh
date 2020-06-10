#!/bin/bash
#
#SBATCH --job-name=qa-albert
#SBATCH --output=logsqa_albert/qa_albert.txt  # output file
#SBATCH -e logsqa_albert/qa_albert.err        # File to which STDERR will be written
#SBATCH --gres=gpu:3
#SBATCH --partition=2080ti-long # Partition to submit to
#SBATCH --mem=40GB
#
#SBATCH --ntasks=1

export SQUAD_DIR='/mnt/nfs/work1/696ds-s20/abajaj/instabase/data/SQUAD2.0'
export ALBERT_MODEL='/mnt/nfs/work1/696ds-s20/abajaj/nlplab/long-term-context/models/albert-base-v2'
export OUTPUT_DIR='/mnt/nfs/work1/696ds-s20/abajaj/instabase/models/qa-albert-squad2.0'

python run_squad.py \
  --model_type albert \
  --model_name_or_path $ALBERT_MODEL \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v2.0.json \
  --predict_file $SQUAD_DIR/dev-v2.0.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 1.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $OUTPUT_DIR \
  --save_steps 1000 \
  --threads 4 \
  --version_2_with_negative


