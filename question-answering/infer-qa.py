# taken from: https://colab.research.google.com/github/spark-ming/albert-qa-demo/blob/master/Question_Answering_with_ALBERT.ipynb

import os
import torch
import time
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import (
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    squad_convert_examples_to_features
)

from transformers.data.processors.squad import SquadResult, SquadV2Processor, SquadExample

from transformers.data.metrics.squad_metrics import compute_predictions_logits

# READER NOTE: Set this flag to use own model, or use pretrained model in the Hugging Face repository

# model_name_or_path = "qa-squad1.0"
model_name_or_path = "bert-large-uncased-whole-word-masking-finetuned-squad" # finetuned checkpoint available directly

output_dir = ""

# Config
n_best_size = 1
max_answer_length = 30
do_lower_case = True
null_score_diff_threshold = 0.0

def to_list(tensor):
    return tensor.detach().cpu().tolist()

# Setup model
config_class, model_class, tokenizer_class = (
    BertConfig, BertForQuestionAnswering, BertTokenizer)
config = config_class.from_pretrained(model_name_or_path)
tokenizer = tokenizer_class.from_pretrained(
    model_name_or_path, do_lower_case=True)
model = model_class.from_pretrained(model_name_or_path, config=config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

processor = SquadV2Processor()

def run_prediction(question_texts, context_text):
    """Setup function to compute predictions"""
    examples = []

    for i, question_text in enumerate(question_texts):
        example = SquadExample(
            qas_id=str(i),
            question_text=question_text,
            context_text=context_text,
            answer_text=None,
            start_position_character=None,
            title="Predict",
            is_impossible=False,
            answers=None,
        )

        examples.append(example)

    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=384,
        doc_stride=128,
        max_query_length=64,
        is_training=False,
        return_dataset="pt",
        threads=1,
    )

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=10)

    all_results = []

    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            example_indices = batch[3]

            outputs = model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs]

                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)
                all_results.append(result)

    output_prediction_file = "predictions.json"
    output_nbest_file = "nbest_predictions.json"
    output_null_log_odds_file = "null_predictions.json"

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        False,  # verbose_logging
        True,  # version_2_with_negative
        null_score_diff_threshold,
        tokenizer,
    )

    return predictions


# context = "New Zealand (Māori: Aotearoa) is a sovereign island country in the southwestern Pacific Ocean. It has a total land area of 268,000 square kilometres (103,500 sq mi), and a population of 4.9 million. New Zealand's capital city is Wellington, and its most populous city is Auckland."
# questions = ["How many people live in New Zealand?",
#              "What's the largest city?"]


data_dir = "/Users/ahsaasbajaj/Documents/Data/CARTA/processed_texts/"
out_dir = '/Users/ahsaasbajaj/Documents/Data/CARTA/answers/qa-squad1.0/'
files = os.listdir(data_dir)
print('Files: ', len(files))

questions = ["Who is incorporating the company?",
             "How many shares are being created?",
             "What are the Common stocks?",
             "What are the Preferred stocks?",
             "What are the Non-cumulative dividends?",
             "What is the Dividend rate per annum per preferred share type?",
             "Number of authorized shares or share class?",
             "Original Issue Price per share?",
             "Liquidation preference or preferred share type?"]

for fname in files:
    fpath = os.path.join(data_dir, fname)
    print('File name: ', fname)
    f = open(fpath)
    context = f.read()

    # Run method
    predictions = run_prediction(questions, context)

    outpath = os.path.join(out_dir, fname)
    outf = open(outpath, 'w')

    # Print results
    for ind, key in enumerate(predictions.keys()):
        print(questions[ind] + " : " + predictions[key])
        print(questions[ind] + " : " + predictions[key], file=outf)

    outf.close()