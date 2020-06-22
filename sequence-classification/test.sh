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

export DATASET=w2  # on which the model is trained
export RANDOM_SEQ_LEN=5  # phrases choosen for None class

# public w2: test split
export MODE=all # all, no-address
export NUM_LABELS=4 # 4, 3
export TEST=./data/${MODE}/${DATASET}-classifier-data-test${RANDOM_SEQ_LEN}.csv
export OUT_DIR=./${DATASET}/${MODE}/${RANDOM_SEQ_LEN}/inference


# public w2: test split
export MODE=no-address # all, no-address
export NUM_LABELS=3 # 4, 3
export TEST=./data/${MODE}/${DATASET}-classifier-data-test${RANDOM_SEQ_LEN}.csv
export OUT_DIR=./${DATASET}/${MODE}/${RANDOM_SEQ_LEN}/inference


# w2 instabase; no-address
export MODE=no-address # all, no-address
export NUM_LABELS=3 # 4, 3
export TEST=./instabase-data/w2-instabase-classifier-no-address.csv
export OUT_DIR=./${DATASET}/${MODE}/${RANDOM_SEQ_LEN}/instabase-data


# resume instabase; no-address
#export MODE=no-address # all, no-address
#export NUM_LABELS=3 # 4, 3
#export TEST=./instabase-data/resume-classifier-no-address.csv
#export OUT_DIR=./${DATASET}/${MODE}/${RANDOM_SEQ_LEN}/instabase-data


# resume instabase; all
export MODE=all # all, no-address
export NUM_LABELS=4 # 4, 3
export TEST=./instabase-data/resume-classifier-all.csv
export OUT_DIR=./${DATASET}/${MODE}/${RANDOM_SEQ_LEN}/instabase-data

export MODEL=./${DATASET}/${MODE}/${RANDOM_SEQ_LEN}/model.pt
export BERT=bert-large-cased

python -u test_classifier.py --modelFile=${MODEL} --testFile=${TEST} --outputDir=${OUT_DIR} --model_name_or_path=${BERT} --num_labels=${NUM_LABELS} --gpu
