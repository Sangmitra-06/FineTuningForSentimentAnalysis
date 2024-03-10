'''
    Qualitative Analysis for the models on tweet dataset
'''
import csv
from datasets import load_dataset
from transformers import AlbertTokenizer, AlbertForSequenceClassification, BertTokenizer, BertForSequenceClassification, \
    DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Loading the emotion dataset
emotion_dataset = load_dataset("tweet_eval","irony", trust_remote_code=True)

# Loading the saved ALBERT tokenizer for the trained model
albert_tokenizer_trained = AlbertTokenizer.from_pretrained("C:/Users/sangm/Onedrive/Desktop/4P84 trial/Albert_tweet")

# Loading the ALBERT model
albert_model = AlbertForSequenceClassification.from_pretrained(
    "C:/Users/sangm/Onedrive/Desktop/4P84 trial/Albert_tweet", num_labels=2)

# Loading the saved BERT tokenizer for the trained model
bert_tokenizer_trained = BertTokenizer.from_pretrained("C:/Users/sangm/Onedrive/Desktop/4P84 trial/Bert_tweet")

# Loading the BERT model
bert_model = BertForSequenceClassification.from_pretrained("C:/Users/sangm/Onedrive/Desktop/4P84 trial/Bert_tweet",
                                                           num_labels=2)

# Loading the saved DistilBERT tokenizer for the trained model
distilbert_tokenizer_trained = DistilBertTokenizer.from_pretrained(
    "C:/Users/sangm/Onedrive/Desktop/4P84 trial/DistilBERT_tweet")

# Loading the DistilBERT model
distilbert_model = DistilBertForSequenceClassification.from_pretrained(
    "C:/Users/sangm/Onedrive/Desktop/4P84 trial/DistilBERT_tweet", num_labels=2)


#  function to tokenize text
def tokenize_text(tokenizer, text):
    tokenized_input = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    return tokenized_input


# function to get prediction for a single example
def get_prediction(model, inputs):
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits).item()
    return predicted_class


# Performing a qualitative analysis for each model
models = {
    "ALBERT (Trained)": (albert_tokenizer_trained, albert_model),
    "BERT (Trained)": (bert_tokenizer_trained, bert_model),
    "DistilBERT (Trained)": (distilbert_tokenizer_trained, distilbert_model),
    "ALBERT (Untrained)": (AlbertTokenizer.from_pretrained('albert-base-v2'),
                           AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=2)),
    "BERT (Untrained)": (BertTokenizer.from_pretrained('bert-base-uncased'),
                         BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)),
    "DistilBERT (Untrained)": (DistilBertTokenizer.from_pretrained('distilbert-base-uncased'),
                               DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                                                   num_labels=2))
}

# Table headers
table_headers = ["Example", "Text", "Actual Label"] + list(models.keys())
table_data = []

# Iterating over examples
for i in range(5):
    example = emotion_dataset["test"][i]
    text = example["text"]
    label_index = example["label"]
    label_name = emotion_dataset["train"].features["label"].int2str(label_index)

    row = [f"Example {i + 1}", text, label_name]

    for _, (tokenizer, model) in models.items():
        tokenized_input = tokenize_text(tokenizer, text)
        predicted_index = get_prediction(model, tokenized_input)
        predicted_name = emotion_dataset["train"].features["label"].int2str(predicted_index)
        row.append(predicted_name)

    table_data.append(row)

# Writing to CSV
with open("tweet_predictions.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(table_headers)
    writer.writerows(table_data)
