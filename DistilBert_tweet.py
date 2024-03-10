'''
    Fine-tuning DistilBERT on the tweet dataset
'''
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, \
    Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score
# function to tokenize dataset
def tokenize_function(example, tokenizer):
    return tokenizer(example["text"], padding="max_length", truncation=True)



#_______________________________________________________________________________________________________________________________________________________________________________
# TASK 1, DATASET PREPROCESSING
#_______________________________________________________________________________________________________________________________________________________________________________

# Loading the tweet data set
dataset_tweet = load_dataset('tweet_eval','irony')
tokenizer_distilBERT = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# tweet dataset tokenized
tokenized_dataset_tweet_distilBERT = dataset_tweet.map(lambda example: tokenize_function(example, tokenizer_distilBERT), batched=True)

#training, testing, and validation split for distilBERT tweet dataset
distilBERT_tweet_train = tokenized_dataset_tweet_distilBERT['train']
distilBERT_tweet_test = tokenized_dataset_tweet_distilBERT['test']
distilBERT_tweet_val = tokenized_dataset_tweet_distilBERT['validation']


#_______________________________________________________________________________________________________________________________________________________________________________
# TASK 2: MODEL TRAINING AND EVALUATION
#_______________________________________________________________________________________________________________________________________________________________________________

num_labels=2
model_DistilBert=DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)

# setting up the calculated hyperparameters
training_args = TrainingArguments(output_dir="./results", # Directory for saving outputs
                                  learning_rate=4.933054930828662e-05, # Learning rate for optimization
                                  per_device_train_batch_size=32, # Batch size for training
                                  per_device_eval_batch_size=16, # Batch size for evaluation
                                  num_train_epochs=3, # Number of training epochs
                                  seed=10, # Seed that will be set at the beginning of training
                                  weight_decay=0.01, # Weight decay for regularization
                                  evaluation_strategy="epoch",
                                  logging_dir="./logs") # Evaluation is done at the end of each epoch
trainer_DistilBERT = Trainer(model=model_DistilBert, args=training_args, train_dataset=distilBERT_tweet_train, eval_dataset=distilBERT_tweet_val, compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))})

eval_results_untrained=trainer_DistilBERT.evaluate(distilBERT_tweet_test)
print(eval_results_untrained)
# fine-tuning the model
trainer_DistilBERT.train()

# Saving the model
trainer_DistilBERT.save_model("C:/Users/sangm/Onedrive/Desktop/4P84 trial/DistilBERT_tweet")
tokenizer_distilBERT.save_pretrained("C:/Users/sangm/Onedrive/Desktop/4P84 trial/DistilBERT_tweet")

# Evaluate the model
eval_results_trained = trainer_DistilBERT.evaluate(distilBERT_tweet_test)
print(eval_results_trained)






