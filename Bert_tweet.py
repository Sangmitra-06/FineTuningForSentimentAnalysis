'''
    Fine-tuning BERT on the tweet dataset
'''
from datasets import load_dataset
from transformers import BertTokenizer, \
    Trainer, TrainingArguments, BertForSequenceClassification
import numpy as np
from sklearn.metrics import accuracy_score

def tokenize_function(example, tokenizer):
    return tokenizer(example["text"], padding="max_length", truncation=True)


#_______________________________________________________________________________________________________________________________________________________________________________
# TASK 1, DATASET PREPROCESSING
#_______________________________________________________________________________________________________________________________________________________________________________

#Loading the tweet data
dataset_tweet = load_dataset('tweet_eval','irony')
tokenizer_BERT = BertTokenizer.from_pretrained('bert-base-uncased')

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
model_BERT_tweet = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
# setting up the calculated hyperparameters
training_args = TrainingArguments(output_dir="./results", # Directory for saving outputs
                                  learning_rate=4.558391424414366e-05, # Learning rate for optimization
                                  per_device_train_batch_size=10, # Batch size for training
                                  per_device_eval_batch_size=16, # Batch size for evaluation
                                  num_train_epochs=5, # Number of training epochs
                                  weight_decay=0.071613350785340314, # Weight decay for regularization
                                  seed=42,  # Seed that will be set at the beginning of training
                                  evaluation_strategy="epoch") # Evaluation is done at the end of each epoch
trainer_BERT_tweet = Trainer(model=model_BERT_tweet, args=training_args, train_dataset=BERT_tweet_train, eval_dataset=BERT_tweet_val, compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))})
eval_results_untrained=trainer_BERT_tweet.evaluate(BERT_tweet_test)
print(eval_results_untrained)
#fine tuning the model
trainer_BERT_tweet.train()
# Saving the model
trainer_BERT_tweet.save_model("C:/Users/sangm/Onedrive/Desktop/4P84 trial/Bert_tweet")
tokenizer_BERT.save_pretrained("C:/Users/sangm/Onedrive/Desktop/4P84 trial/Bert_tweet")
#Evaluate the model
eval_results_trained = trainer_BERT_tweet.evaluate(BERT_tweet_test)
print(eval_results_trained)






