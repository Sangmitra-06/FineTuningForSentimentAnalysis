'''
    Fine-tuning DistilBERT on the emotion dataset
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

#Loading the emotion data set
dataset_emotion = load_dataset('dair-ai/emotion')
tokenizer_distilBERT = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# emotion dataset tokenized
tokenized_dataset_emotion_distilBERT = dataset_emotion.map(lambda example: tokenize_function(example, tokenizer_distilBERT), batched=True)

#training, testing, and validation split for distilBERT emotion dataset
distilBERT_emotion_train = tokenized_dataset_emotion_distilBERT['train']
distilBERT_emotion_test = tokenized_dataset_emotion_distilBERT['test']
distilBERT_emotion_val = tokenized_dataset_emotion_distilBERT['validation']


#_______________________________________________________________________________________________________________________________________________________________________________
# TASK 2: MODEL TRAINING AND EVALUATION
#_______________________________________________________________________________________________________________________________________________________________________________

num_labels=6
model_DistilBert=DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
# setting up the calculated hyperparameters
training_args = TrainingArguments(output_dir="./results", # Directory for saving outputs
                                  learning_rate=9.142655848539377e-06, # Learning rate for optimization
                                  per_device_train_batch_size=16, # Batch size for training
                                  per_device_eval_batch_size=16, # Batch size for evaluation
                                  num_train_epochs=4, # Number of training epochs
                                  weight_decay=0.01, # Weight decay for regularization
                                  seed=19, # Seed that will be set at the beginning of training
                                  evaluation_strategy="epoch",
                                  logging_dir="./logs") # Evaluation is done at the end of each epoch
trainer_DistilBERT = Trainer(model=model_DistilBert, args=training_args, train_dataset=distilBERT_emotion_train, eval_dataset=distilBERT_emotion_val, compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))})

eval_results_untrained=trainer_DistilBERT.evaluate(distilBERT_emotion_test)
print(eval_results_untrained)
# fine tuning the model
trainer_DistilBERT.train()

# Saving the model
trainer_DistilBERT.save_model("C:/Users/sangm/Onedrive/Desktop/4P84 trial/DistilBERT_emotion")
tokenizer_distilBERT.save_pretrained("C:/Users/sangm/Onedrive/Desktop/4P84 trial/DistilBERT_emotion")

# Evaluate the model
eval_results_trained = trainer_DistilBERT.evaluate(distilBERT_emotion_test)
print(eval_results_trained)






