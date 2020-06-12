#!/bin/bash
#!/bin/bash
#
#SBATCH --job-name=bert-cls
#SBATCH --output=logscls/infer_class_%j.txt  # output file
#SBATCH -e logscls/infer_class_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=2080ti-short # Partition to submit to
#SBATCH --mem=40GB
#
#SBATCH --ntasks=1

export DATASET=w2
export RANDOM_SEQ_LEN=3

export MODEL=./${DATASET}/${RANDOM_SEQ_LEN}/model.pt

export TEST=./data/${DATASET}-classifier-data${RANDOM_SEQ_LEN}-test.csv
export OUT_DIR=./${DATASET}/${RANDOM_SEQ_LEN}  # always make new directory, since code empties OUT_DIR and then writes

python -u test_classifier.py --modelFile=${MODEL} --testFile=${TEST} --outputDir=${OUT_DIR} --gpu
