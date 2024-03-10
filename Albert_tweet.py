'''
    Fine-tuning ALBERT on the tweet dataset
'''
from datasets import load_dataset
from transformers import AlbertTokenizer, \
    Trainer, TrainingArguments, AlbertForSequenceClassification
import numpy as np
from sklearn.metrics import accuracy_score
# function to tokenize dataset
def tokenize_function(example, tokenizer):
    return tokenizer(example["text"], padding="max_length", truncation=True)


#_______________________________________________________________________________________________________________________________________________________________________________
# TASK 1, DATASET PREPROCESSING
#_______________________________________________________________________________________________________________________________________________________________________________

#Loading the tweet data set
dataset_tweet = load_dataset('tweet_eval', 'irony')
tokenizer_ALBERT = AlbertTokenizer.from_pretrained('albert-base-v2')


# tweet eval dataset tokenized
tokenized_dataset_tweet_ALBERT = dataset_tweet.map(lambda example: tokenize_function(example, tokenizer_ALBERT), batched=True)

#training, testing, and validation split for ALBERT tweet dataset
ALBERT_tweet_train = tokenized_dataset_tweet_ALBERT['train']
ALBERT_tweet_test = tokenized_dataset_tweet_ALBERT['test']
ALBERT_tweet_val = tokenized_dataset_tweet_ALBERT['validation']


#_______________________________________________________________________________________________________________________________________________________________________________
# TASK 2: MODEL TRAINING AND EVALUATION
#_______________________________________________________________________________________________________________________________________________________________________________

num_labels=2
model_ALBERT_tweet = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=num_labels)
# setting up the calculated hyperparameters
training_args = TrainingArguments(output_dir="./results", # Directory for saving outputs
                                  learning_rate=1.2957832928572053e-06, # Learning rate for optimization
                                  per_device_train_batch_size=4, # Batch size for training
                                  per_device_eval_batch_size=16, # Batch size for evaluation
                                  num_train_epochs=5, # Number of training epochs
                                  seed=42, # Seed that will be set at the beginning of training
                                  weight_decay=0.01, # Weight decay for regularization
                                  evaluation_strategy="epoch") # Evaluation is done at the end of each epoch
trainer_ALBERT_tweet = Trainer(model=model_ALBERT_tweet, args=training_args, train_dataset=ALBERT_tweet_train, eval_dataset=ALBERT_tweet_val, compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))})
eval_results_untrained = trainer_ALBERT_tweet.evaluate(ALBERT_tweet_test)
print(eval_results_untrained)
# fine-tuning the model
trainer_ALBERT_tweet.train()
# Saving the model
trainer_ALBERT_tweet.save_model("C:/Users/sangm/Onedrive/Desktop/4P84 trial/Albert_tweet")
tokenizer_ALBERT.save_pretrained("C:/Users/sangm/Onedrive/Desktop/4P84 trial/Albert_tweet")
#Evaluate the model
eval_results_trained = trainer_ALBERT_tweet.evaluate(ALBERT_tweet_test)
print(eval_results_trained)






