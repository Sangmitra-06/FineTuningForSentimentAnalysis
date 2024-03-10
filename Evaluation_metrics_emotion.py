'''
    Collecting the evaluation metrics for ALBERT, BERT, and DistilBERT for the emotion dataset
'''
from datasets import load_dataset
from transformers import BertForSequenceClassification, DistilBertForSequenceClassification, \
    AlbertForSequenceClassification, Trainer,  BertTokenizer, DistilBertTokenizer, AlbertTokenizer
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd

# Function to tokenize examples
def tokenize_function(example, tokenizer):
    return tokenizer(example["text"], padding="max_length", truncation=True)
# Loading the emotion dataset
dataset = load_dataset('dair-ai/emotion')

# Loading tokenizers
tokenizer_BERT = BertTokenizer.from_pretrained("C:/Users/sangm/Onedrive/Desktop/4P84 trial/Bert_emotion")
tokenizer_DistilBERT = DistilBertTokenizer.from_pretrained("C:/Users/sangm/Onedrive/Desktop/4P84 trial/DistilBERT_emotion")
tokenizer_ALBERT = AlbertTokenizer.from_pretrained("C:/Users/sangm/Onedrive/Desktop/4P84 trial/ALBERT_emotion")

# Tokenizing the dataset for each model
tokenized_dataset_BERT = dataset.map(lambda example: tokenize_function(example, tokenizer_BERT), batched=True)
tokenized_dataset_DistilBERT = dataset.map(lambda example: tokenize_function(example, tokenizer_DistilBERT), batched=True)
tokenized_dataset_ALBERT = dataset.map(lambda example: tokenize_function(example, tokenizer_ALBERT), batched=True)

# Splitting the dataset into train, test, and validation sets
train_dataset = tokenized_dataset_BERT['train']
test_dataset = tokenized_dataset_BERT['test']
val_dataset = tokenized_dataset_BERT['validation']
# Defining the untrained models
model_BERT_untrained = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
model_DistilBERT_untrained = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6)
model_ALBERT_untrained = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=6)

# Defining the fine-tuned models
model_BERT_finetuned = BertForSequenceClassification.from_pretrained('C:/Users/sangm/Onedrive/Desktop/4P84 trial/Bert_emotion', num_labels=6)
model_DistilBERT_finetuned = DistilBertForSequenceClassification.from_pretrained('C:/Users/sangm/Onedrive/Desktop/4P84 trial/DistilBERT_emotion', num_labels=6)
model_ALBERT_finetuned = AlbertForSequenceClassification.from_pretrained('C:/Users/sangm/Onedrive/Desktop/4P84 trial/ALBERT_emotion', num_labels=6)

# Defining the Trainers for untrained models
trainer_BERT_untrained = Trainer(model=model_BERT_untrained,
                                 compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))})

trainer_DistilBERT_untrained = Trainer(model=model_DistilBERT_untrained,
                                       compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))})

trainer_ALBERT_untrained = Trainer(model=model_ALBERT_untrained,
                                   compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))})

# Defining the Trainers for fine-tuned models
trainer_BERT_finetuned = Trainer(model=model_BERT_finetuned,
                                 compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))})

trainer_DistilBERT_finetuned = Trainer(model=model_DistilBERT_finetuned,
                                       compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))})

trainer_ALBERT_finetuned = Trainer(model=model_ALBERT_finetuned,
                                   compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))})

# Evaluating the untrained models on the train, test, and validation data
eval_results_train_untrained_BERT = trainer_BERT_untrained.evaluate(train_dataset)
eval_results_val_untrained_BERT = trainer_BERT_untrained.evaluate(val_dataset)
eval_results_test_untrained_BERT = trainer_BERT_untrained.evaluate(test_dataset)

eval_results_train_untrained_DistilBERT = trainer_DistilBERT_untrained.evaluate(train_dataset)
eval_results_val_untrained_DistilBERT = trainer_DistilBERT_untrained.evaluate(val_dataset)
eval_results_test_untrained_DistilBERT = trainer_DistilBERT_untrained.evaluate(test_dataset)

eval_results_train_untrained_ALBERT = trainer_ALBERT_untrained.evaluate(train_dataset)
eval_results_val_untrained_ALBERT = trainer_ALBERT_untrained.evaluate(val_dataset)
eval_results_test_untrained_ALBERT = trainer_ALBERT_untrained.evaluate(test_dataset)

# Evaluating the fine-tuned models on the train, test, and validation data
eval_results_train_finetuned_BERT = trainer_BERT_finetuned.evaluate(train_dataset)
eval_results_val_finetuned_BERT = trainer_BERT_finetuned.evaluate(val_dataset)
eval_results_test_finetuned_BERT = trainer_BERT_finetuned.evaluate(test_dataset)

eval_results_train_finetuned_DistilBERT = trainer_DistilBERT_finetuned.evaluate(train_dataset)
eval_results_val_finetuned_DistilBERT = trainer_DistilBERT_finetuned.evaluate(val_dataset)
eval_results_test_finetuned_DistilBERT = trainer_DistilBERT_finetuned.evaluate(test_dataset)

eval_results_train_finetuned_ALBERT = trainer_ALBERT_finetuned.evaluate(train_dataset)
eval_results_val_finetuned_ALBERT = trainer_ALBERT_finetuned.evaluate(val_dataset)
eval_results_test_finetuned_ALBERT = trainer_ALBERT_finetuned.evaluate(test_dataset)

# Printing the results in a table format
print("Table 1: Model Performance")
print("BERT Model:")
print("Untrained:")
print("Train Accuracy:", eval_results_train_untrained_BERT["eval_accuracy"])
print("Validation Accuracy:", eval_results_val_untrained_BERT["eval_accuracy"])
print("Test Accuracy:", eval_results_test_untrained_BERT["eval_accuracy"])
print("Fine-tuned:")
print("Train Accuracy:", eval_results_train_finetuned_BERT["eval_accuracy"])
print("Validation Accuracy:", eval_results_val_finetuned_BERT["eval_accuracy"])
print("Test Accuracy:", eval_results_test_finetuned_BERT["eval_accuracy"])

print("\nDistilBERT Model:")
print("Untrained:")
print("Train Accuracy:", eval_results_train_untrained_DistilBERT["eval_accuracy"])
print("Validation Accuracy:", eval_results_val_untrained_DistilBERT["eval_accuracy"])
print("Test Accuracy:", eval_results_test_untrained_DistilBERT["eval_accuracy"])
print("Fine-tuned:")
print("Train Accuracy:", eval_results_train_finetuned_DistilBERT["eval_accuracy"])
print("Validation Accuracy:", eval_results_val_finetuned_DistilBERT["eval_accuracy"])
print("Test Accuracy:", eval_results_test_finetuned_DistilBERT["eval_accuracy"])

print("\nALBERT Model:")
print("Untrained:")
print("Train Accuracy:", eval_results_train_untrained_ALBERT["eval_accuracy"])
print("Validation Accuracy:", eval_results_val_untrained_ALBERT["eval_accuracy"])
print("Test Accuracy:", eval_results_test_untrained_ALBERT["eval_accuracy"])
print("Fine-tuned:")
print("Train Accuracy:", eval_results_train_finetuned_ALBERT["eval_accuracy"])
print("Validation Accuracy:", eval_results_val_finetuned_ALBERT["eval_accuracy"])
print("Test Accuracy:", eval_results_test_finetuned_ALBERT["eval_accuracy"])



# Creating a DataFrame for BERT model performance
df_bert_untrained = pd.DataFrame({
    "Model State": ["Untrained", "Untrained", "Untrained", "Fine-tuned", "Fine-tuned", "Fine-tuned"],
    "Dataset": ["Train", "Validation", "Test"]*2,
    "Accuracy": [eval_results_train_untrained_BERT["eval_accuracy"], eval_results_val_untrained_BERT["eval_accuracy"], eval_results_test_untrained_BERT["eval_accuracy"],
                 eval_results_train_finetuned_BERT["eval_accuracy"], eval_results_val_finetuned_BERT["eval_accuracy"], eval_results_test_finetuned_BERT["eval_accuracy"]]
})

# Creating a DataFrame for DistilBERT model performance
df_distilbert_untrained = pd.DataFrame({
    "Model State": ["Untrained", "Untrained", "Untrained", "Fine-tuned", "Fine-tuned", "Fine-tuned"],
    "Dataset": ["Train", "Validation", "Test"]*2,
    "Accuracy": [eval_results_train_untrained_DistilBERT["eval_accuracy"], eval_results_val_untrained_DistilBERT["eval_accuracy"], eval_results_test_untrained_DistilBERT["eval_accuracy"],
                 eval_results_train_finetuned_DistilBERT["eval_accuracy"], eval_results_val_finetuned_DistilBERT["eval_accuracy"], eval_results_test_finetuned_DistilBERT["eval_accuracy"]]
})

# Creating a DataFrame for ALBERT model performance
df_albert_untrained = pd.DataFrame({
    "Model State": ["Untrained", "Untrained", "Untrained", "Fine-tuned", "Fine-tuned", "Fine-tuned"],
    "Dataset": ["Train", "Validation", "Test"]*2,
    "Accuracy": [eval_results_train_untrained_ALBERT["eval_accuracy"], eval_results_val_untrained_ALBERT["eval_accuracy"], eval_results_test_untrained_ALBERT["eval_accuracy"],
                 eval_results_train_finetuned_ALBERT["eval_accuracy"], eval_results_val_finetuned_ALBERT["eval_accuracy"], eval_results_test_finetuned_ALBERT["eval_accuracy"]]
})

# Combining all the DataFrames for all models
df_all_models = pd.concat([df_bert_untrained, df_distilbert_untrained, df_albert_untrained], keys=['BERT', 'DistilBERT', 'ALBERT'], names=['Model'])

# Printing the combined DataFrame
print("Table 1: Model Performance")
print(df_all_models)

