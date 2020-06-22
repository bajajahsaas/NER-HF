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

export DATASET=arithmetic  # on which the model is trained
export NUM_LABELS=2
export TEST=./data/${DATASET}-test.csv
export OUT_DIR=./${DATASET}/results

export MODEL=./${DATASET}/model.pt
export BERT=bert-large-cased

python -u test_classifier.py --modelFile=${MODEL} --testFile=${TEST} --outputDir=${OUT_DIR} --model_name_or_path=${BERT} --num_labels=${NUM_LABELS} --gpu
