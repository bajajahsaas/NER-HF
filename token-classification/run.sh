# GERMEVAL 2014 (german NER)

# curl -L 'https://sites.google.com/site/germeval2014ner/data/NER-de-train.tsv?attredirects=0&d=1' \
# | grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > train.txt.tmp
# curl -L 'https://sites.google.com/site/germeval2014ner/data/NER-de-dev.tsv?attredirects=0&d=1' \
# | grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > dev.txt.tmp
# curl -L 'https://sites.google.com/site/germeval2014ner/data/NER-de-test.tsv?attredirects=0&d=1' \
# | grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > test.txt.tmp
# export MAX_LENGTH=128
# export BERT_MODEL=bert-base-multilingual-cased
# python3 preprocess.py train.txt.tmp $BERT_MODEL $MAX_LENGTH > train.txt
# python3 preprocess.py dev.txt.tmp $BERT_MODEL $MAX_LENGTH > dev.txt
# python3 preprocess.py test.txt.tmp $BERT_MODEL $MAX_LENGTH > test.txt
# cat train.txt dev.txt test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > labels.txt

# WNUT’17 (English NER)

mkdir -p data_wnut_17

curl -L 'https://github.com/leondz/emerging_entities_17/raw/master/wnut17train.conll'  | tr '\t' ' ' > data_wnut_17/train.txt.tmp
curl -L 'https://github.com/leondz/emerging_entities_17/raw/master/emerging.dev.conll' | tr '\t' ' ' > data_wnut_17/dev.txt.tmp
curl -L 'https://raw.githubusercontent.com/leondz/emerging_entities_17/master/emerging.test.annotated' | tr '\t' ' ' > data_wnut_17/test.txt.tmp
export MAX_LENGTH=128
export BERT_MODEL=bert-large-cased

python3 preprocess.py data_wnut_17/train.txt.tmp $BERT_MODEL $MAX_LENGTH > data_wnut_17/train.txt
python3 preprocess.py data_wnut_17/dev.txt.tmp $BERT_MODEL $MAX_LENGTH > data_wnut_17/dev.txt
python3 preprocess.py data_wnut_17/test.txt.tmp $BERT_MODEL $MAX_LENGTH > data_wnut_17/test.txt

cat data_wnut_17/train.txt data_wnut_17/dev.txt data_wnut_17/test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > data_wnut_17/labels.txt