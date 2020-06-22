import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import trange
import pandas as pd
import numpy as np
from data_utils import readData, flat_accuracy, save_plots_models
import argparse
import os, sys

sys.path.insert(0, os.path.abspath('..'))
os.environ["PYTHONIOENCODING"] = "utf-8"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help='If set, use the GPU')
    parser.add_argument('--trainFile', type=str, default='',
                        help='Folder path to read training input')
    parser.add_argument('--devFile', type=str, default='',
                        help='Folder path to read validation input')
    parser.add_argument('--testFile', type=str, default='',
                        help='Folder path to read testing input')
    parser.add_argument('--outputDir', type=str, default='',
                        help='Folder path to write output to')
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--num_labels", default=None, type=int, required=True,
                        help="Number of labels in classification")
    args = parser.parse_args()

outDir = args.outputDir
trainFile = args.trainFile
devFile = args.devFile
testFile = args.testFile

print('outDir: {0}, trainFile: {1}, devFile: {2}, testFile: {3}'.format(outDir, trainFile, devFile, testFile))

if os.path.exists(outDir):
    filelist = [f for f in os.listdir(outDir)]
    for f in filelist:
        os.remove(os.path.join(outDir, f))
else:
    os.makedirs(outDir)

device = torch.device("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")
print('Device', device)
n_gpu = torch.cuda.device_count()

model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=args.num_labels)
tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=True)

model.cuda()

train_inputs, train_labels, train_masks = readData(tokenizer, args, mode="train")
validation_inputs, validation_labels, validation_masks = readData(tokenizer, args, mode="dev")

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

# Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
batch_size = 32

# Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
# with an iterator the entire dataset does not need to be loaded into memory

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

'''
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]
optimizer = BertAdam(optimizer_grouped_parameters,lr=2e-5,warmup=.1)
'''
# Check BertAdam: https://github.com/huggingface/transformers
# Migration: https://huggingface.co/transformers/migration.html

lr = 2e-5
max_grad_norm = 1.0
num_training_steps = 1000
num_warmup_steps = 100
warmup_proportion = float(num_warmup_steps) / float(num_training_steps)  # 0.1

optimizer = AdamW(model.parameters(), lr=lr,
                  correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                            num_training_steps=num_training_steps)  # PyTorch scheduler

train_loss_set = []
train_acc_set = []
val_acc_set = []

# Number of training epochs (authors recommend between 2 and 4)
epochs = 100

# trange is a tqdm wrapper around the normal python range
for epoch in trange(epochs, desc="Epoch"):
    print("########################")
    print('Training for {0} epoch'.format(epoch + 1))
    print("########################")
    # Set our model to training mode (as opposed to evaluation mode)
    model.train()

    # Tracking variables
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    train_accuracy = 0

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        model.train()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        # tuple of float and tensor
        loss, logits = outputs[:2]
        # train_loss_set.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Update tracking variable
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_train_accuracy = flat_accuracy(logits, label_ids)
        train_accuracy += tmp_train_accuracy

        if step % 100 == 0:
            print('Training for {0} step'.format(step + 1))
            train_loss_set.append(tr_loss * 1.0 / nb_tr_steps)
            train_acc_set.append(train_accuracy * 1.0 / nb_tr_steps)

    print('Total tr steps: {0}, total tr examples: {1}'.format(nb_tr_steps, nb_tr_examples))
    print("Train loss: {0:0.4f}".format(tr_loss / nb_tr_steps))
    print("Training Accuracy: {0:0.4f}".format(train_accuracy / nb_tr_steps))

    train_loss_set.append(tr_loss * 1.0 / nb_tr_steps)
    train_acc_set.append(train_accuracy * 1.0 / nb_tr_steps)

    # Validation

    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        # returns tuple with single tensor i.e. logits
        logits = outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("Validation Accuracy: {0:0.4f}".format(eval_accuracy / nb_eval_steps))
    val_acc_set.append(eval_accuracy * 1.0 / nb_eval_steps)

    if (epoch + 1) % 10 == 0:
        save_plots_models(outDir, train_loss_set, train_acc_set, val_acc_set, model.state_dict(), epoch, epochs)

# dump final plots and models
save_plots_models(outDir, train_loss_set, train_acc_set, val_acc_set, model.state_dict(), epochs, epochs)


# testing
input_ids, labels, attention_masks = readData(tokenizer, args, mode="test")

prediction_inputs = torch.tensor(input_ids)
prediction_masks = torch.tensor(attention_masks)
prediction_labels = torch.tensor(labels)

batch_size = 32
prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

model.eval()

# Tracking variables
predictions, true_labels = [], []
nb_eval_steps = 0
eval_accuracy = 0
csv_output = []

# Predict
for batch in prediction_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    # Telling the model not to compute or store gradients, saving memory and speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    logits = outputs[0]
    # shape (batch_size, config.num_labels)
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    # Store predictions and true labels
    predictions.append(logits)
    true_labels.append(label_ids)
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1
    pred_flat = np.argmax(logits, axis=1).flatten()
    labels_flat = label_ids.flatten()
    count = 0
    for i in range(pred_flat.shape[0]):
        # iterate over the batch
        csv_output.append((b_input_ids[i], pred_flat[i], labels_flat[i]))

print('Test Accuracy Accuracy: {0:0.4f}'.format((float(eval_accuracy) / float(nb_eval_steps))))

flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
flat_true_labels = [item for sublist in true_labels for item in sublist]

labels = [x for x in range(args.num_labels)]

micro_precision = precision_score(flat_true_labels, flat_predictions)
micro_recall = recall_score(flat_true_labels, flat_predictions)
micro_f1 = f1_score(flat_true_labels, flat_predictions)
print('Test R: {0:0.4f}, P: {1:0.4f}, F1: {2:0.4f}'.format(micro_recall, micro_precision, micro_f1))

testFile = args.testFile.split('/')
testFile = testFile[len(testFile) - 1]  # take the last part

outFile = outDir + "/" + testFile[:len(testFile) - 4] + '_output.txt'
file = open(outFile, 'w')
print('Micro Test R: {0:0.4f}, P: {1:0.4f}, F1: {2:0.4f}'.format(micro_recall, micro_precision, micro_f1), file=file)

print('Saving scores to: ', outFile)

headings = ['context', 'predicted', 'label']
df = pd.DataFrame(columns=headings)
for ids, pred, label in csv_output:
    ids = np.trim_zeros(ids.cpu().numpy())
    sentence = tokenizer.convert_ids_to_tokens(ids)[1:-1]
    # data = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]
    data = [{'context': ' '.join(sentence), 'predicted': str(pred), 'label': str(label)}]
    df = df.append(pd.DataFrame(data, columns=headings))


outFile = outDir + "/" + testFile[:len(testFile) - 4] + '_output.csv'
df.to_csv(outFile, index=False)
print('Saving output file to: ', outFile)

print('Process Completed')