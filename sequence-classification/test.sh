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

# TRAINING CONFIG
######################
export DATASET=w2  # on which the model is trained
export RANDOM_SEQ_LEN=5  # phrases len (w2 training) or x_thteshold (public training) chosen for None class
########################


# INFERENCE DATA
########################
# public w2: test split
export MODE=all # all, no-address
export NUM_LABELS=4 # 4, 3
export TEST=./data/${MODE}/${DATASET}-classifier-data-test${RANDOM_SEQ_LEN}.csv
export OUT_DIR=./${DATASET}/${MODE}/${RANDOM_SEQ_LEN}/inference

# w2 instabase: no-address
export TEST_DATA=w2-instabase
export MODE=no-address
export NUM_LABELS=3

# resume instabase: no-address
#export TEST_DATA=resume
#export MODE=no-address
#export NUM_LABELS=3

# resume instabase: all
#export TEST_DATA=resume
#export MODE=all
#export NUM_LABELS=4

#########################


export TEST=../../../Data/${TEST_DATA}/dataset/${TEST_DATA}-classifier-${MODE}.csv
export OUT_DIR=./${DATASET}/${MODE}/${RANDOM_SEQ_LEN}/instabase-data
export MODEL=./${DATASET}/${MODE}/${RANDOM_SEQ_LEN}/model.pt
export BERT=bert-large-cased
python -u test_classifier.py --modelFile=${MODEL} --testFile=${TEST} --outputDir=${OUT_DIR} --model_name_or_path=${BERT} --num_labels=${NUM_LABELS} --gpu
