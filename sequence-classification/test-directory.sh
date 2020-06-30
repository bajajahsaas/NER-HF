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

export DATASET=public  # w2, public: on which the model is trained
export RANDOM_SEQ_LEN=200  # 5, 200: phrases choosen for None class

export MODE=no-address
export NUM_LABELS=3

export MODEL=./${DATASET}/${MODE}/${RANDOM_SEQ_LEN}/model.pt
export BERT=bert-large-cased

export TEST_DATA=resume # w2-instabase, resume
export X_DIST_THRESHOLD=100 # 200, 100

export TEST_DIR=../../../Data/${TEST_DATA}/ocr/phrases_${X_DIST_THRESHOLD}_bert
export OUT_DIR=${TEST_DIR}/outputs

for file in ${TEST_DIR}/*csv; do
  echo "$file";
  python -u test_classifier.py --modelFile=${MODEL} --testFile=${file} --outputDir=${OUT_DIR} --model_name_or_path=${BERT} --num_labels=${NUM_LABELS} --gpu
done