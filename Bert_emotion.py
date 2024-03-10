'''
    Fine-tuning BERT on the emotion dataset
'''
from datasets import load_dataset
from transformers import  BertTokenizer, \
    Trainer, TrainingArguments, BertForSequenceClassification
import numpy as np
from sklearn.metrics import accuracy_score

# function to tokenize dataset
def tokenize_function(example, tokenizer):
    return tokenizer(example["text"], padding="max_length", truncation=True)

#_______________________________________________________________________________________________________________________________________________________________________________
# TASK 1, DATASET PREPROCESSING
#_______________________________________________________________________________________________________________________________________________________________________________

# Loading the emotion data set
dataset_emotion = load_dataset('dair-ai/emotion')
tokenizer_BERT = BertTokenizer.from_pretrained('bert-base-uncased')

# emotion dataset tokenized
tokenized_dataset_emotion_BERT = dataset_emotion.map(lambda example: tokenize_function(example, tokenizer_BERT), batched=True)

#training, testing, and validation split for BERT emotion dataset
BERT_emotion_train = tokenized_dataset_emotion_BERT['train']
BERT_emotion_test = tokenized_dataset_emotion_BERT['test']
BERT_emotion_val = tokenized_dataset_emotion_BERT['validation']


#_______________________________________________________________________________________________________________________________________________________________________________
# TASK 2: MODEL TRAINING AND EVALUATION
#_______________________________________________________________________________________________________________________________________________________________________________
num_labels=6

model_BERT_emotion=BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# setting up the calculated hyperparameters
training_args = TrainingArguments(output_dir="./results", # Directory for saving outputs
                                  learning_rate=2.4730814834253496e-05, # Learning rate for optimization
                                  per_device_train_batch_size=4, # Batch size for training
                                  per_device_eval_batch_size=16, # Batch size for evaluation
                                  num_train_epochs=2,# Number of training epochs
                                  seed=4, # Seed that will be set at the beginning of training
                                  weight_decay=0.01, # Weight decay for regularization
                                  evaluation_strategy="epoch") # Evaluation is done at the end of each epoch
trainer_BERT_emotion = Trainer(model=model_BERT_emotion, args=training_args, train_dataset=BERT_emotion_train, eval_dataset=BERT_emotion_val, compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))})

eval_results_untrained=trainer_BERT_emotion.evaluate(BERT_emotion_test)
print(eval_results_untrained)

# fine-tuning the model
trainer_BERT_emotion.train()

# Saving the model
trainer_BERT_emotion.save_model("C:/Users/sangm/Onedrive/Desktop/4P84 trial/Bert_emotion")
tokenizer_BERT.save_pretrained("C:/Users/sangm/Onedrive/Desktop/4P84 trial/Bert_emotion")

# Evaluate the model
eval_results_trained = trainer_BERT_emotion.evaluate(BERT_emotion_test)
print(eval_results_trained)







