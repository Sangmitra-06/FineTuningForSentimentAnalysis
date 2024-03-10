'''
    Generating confusion matrices for BERT, DistilBERT, and ALBERT on tweet dataset
'''
from datasets import load_dataset
from transformers import AlbertTokenizer, DistilBertTokenizer, BertTokenizer, AlbertForSequenceClassification, \
    DistilBertForSequenceClassification, BertForSequenceClassification, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# function to tokenize the dataset
def tokenize_function(example, tokenizer):
    return tokenizer(example["text"], padding="max_length", truncation=True)

# Loading the tweet dataset
dataset_tweet = load_dataset('tweet_eval', 'irony')
true_labels = dataset_tweet["test"]["label"]

# Initializing the tokenizers and models
tokenizer_BERT = BertTokenizer.from_pretrained('C:/Users/sangm/Onedrive/Desktop/4P84 trial/Bert_tweet')
tokenizer_DistilBERT = DistilBertTokenizer.from_pretrained('C:/Users/sangm/Onedrive/Desktop/4P84 trial/DistilBERT_tweet')
tokenizer_ALBERT = AlbertTokenizer.from_pretrained('C:/Users/sangm/Onedrive/Desktop/4P84 trial/Albert_tweet')

model_BERT = BertForSequenceClassification.from_pretrained('C:/Users/sangm/Onedrive/Desktop/4P84 trial/Bert_tweet')
model_DistilBERT = DistilBertForSequenceClassification.from_pretrained('C:/Users/sangm/Onedrive/Desktop/4P84 trial/DistilBERT_tweet')
model_ALBERT = AlbertForSequenceClassification.from_pretrained('C:/Users/sangm/Onedrive/Desktop/4P84 trial/Albert_tweet')

# Tokenizing the tweet dataset for BERT, DistilBERT, and ALBERT
tokenized_dataset_tweet_BERT = dataset_tweet.map(lambda example: tokenize_function(example, tokenizer_BERT), batched=True)
tokenized_dataset_tweet_DistilBERT = dataset_tweet.map(lambda example: tokenize_function(example, tokenizer_DistilBERT), batched=True)
tokenized_dataset_tweet_ALBERT = dataset_tweet.map(lambda example: tokenize_function(example, tokenizer_ALBERT), batched=True)

# Getting the test datasets for BERT, DistilBERT, and ALBERT
BERT_tweet_test = tokenized_dataset_tweet_BERT['test']
DistilBERT_tweet_test = tokenized_dataset_tweet_DistilBERT['test']
ALBERT_tweet_test = tokenized_dataset_tweet_ALBERT['test']

# Initializing trainers for BERT, DistilBERT, and ALBERT
trainer_BERT = Trainer(model=model_BERT,
                        compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))})
trainer_DistilBERT = Trainer(model=model_DistilBERT,
                             compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))})
trainer_ALBERT = Trainer(model=model_ALBERT,
                         compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))})

# Generating predictions on the test sets for BERT, DistilBERT, and ALBERT
predictions_BERT = trainer_BERT.predict(BERT_tweet_test)
predictions_DistilBERT = trainer_DistilBERT.predict(DistilBERT_tweet_test)
predictions_ALBERT = trainer_ALBERT.predict(ALBERT_tweet_test)

# Getting the predicted labels for BERT, DistilBERT, and ALBERT
predicted_labels_BERT = np.argmax(predictions_BERT.predictions, axis=1)
predicted_labels_DistilBERT = np.argmax(predictions_DistilBERT.predictions, axis=1)
predicted_labels_ALBERT = np.argmax(predictions_ALBERT.predictions, axis=1)

# Generating confusion matrices for BERT, DistilBERT, and ALBERT
confusion_mat_BERT = confusion_matrix(true_labels, predicted_labels_BERT)
confusion_mat_DistilBERT = confusion_matrix(true_labels, predicted_labels_DistilBERT)
confusion_mat_ALBERT = confusion_matrix(true_labels, predicted_labels_ALBERT)

# Plotting the confusion matrix for BERT
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat_BERT, annot=True, fmt='d', cmap='Purples', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for BERT')
plt.show()

# Plotting the confusion matrix for DistilBERT
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat_DistilBERT, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for DistilBERT')
plt.show()

# Plotting the confusion matrix for ALBERT
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat_ALBERT, annot=True, fmt='d', cmap='Reds', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for ALBERT')
plt.show()
