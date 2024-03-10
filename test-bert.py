from datasets import load_dataset
from transformers import AlbertTokenizer, AlbertForSequenceClassification, \
    Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score

def tokenize_function(example, tokenizer):
    return tokenizer(example["text"], padding="max_length", truncation=True)


#_______________________________________________________________________________________________________________________________________________________________________________
# TASK 1, DATASET PREPROCESSING
#_______________________________________________________________________________________________________________________________________________________________________________

#Loading the tweet data
dataset_tweet = load_dataset('tweet_eval','irony')
tokenizer_BERT = AlbertTokenizer.from_pretrained('Sangmitra-06/Albert_tweet')

# tweet dataset tokenized
tokenized_dataset_tweet_BERT = dataset_tweet.map(lambda example: tokenize_function(example, tokenizer_BERT), batched=True)


#training, testing, and validation split for BERT tweet dataset
BERT_tweet_train = tokenized_dataset_tweet_BERT['train']
BERT_tweet_test = tokenized_dataset_tweet_BERT['test']
BERT_tweet_val = tokenized_dataset_tweet_BERT['validation']


#_______________________________________________________________________________________________________________________________________________________________________________
# TASK 2: MODEL TRAINING AND EVALUATION
#_______________________________________________________________________________________________________________________________________________________________________________

num_labels=2
model_BERT_tweet = AlbertForSequenceClassification.from_pretrained('Sangmitra-06/Albert_tweet', num_labels=num_labels)
trainer_BERT_tweet = Trainer(model=model_BERT_tweet, compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))})
eval_results_untrained=trainer_BERT_tweet.evaluate(BERT_tweet_test)
print(eval_results_untrained)