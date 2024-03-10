'''
    Fine-tuning ALBERT on the emotion dataset
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

# Loading the emotion dataset
dataset_emotion = load_dataset('dair-ai/emotion')
tokenizer_ALBERT = AlbertTokenizer.from_pretrained('albert-base-v2')

# emotion dataset tokenized
tokenized_dataset_emotion_ALBERT = dataset_emotion.map(lambda example: tokenize_function(example, tokenizer_ALBERT), batched=True)


# training, testing, and validation split for ALBERT emotion dataset
ALBERT_emotion_train = tokenized_dataset_emotion_ALBERT['train']
ALBERT_emotion_test = tokenized_dataset_emotion_ALBERT['test']
ALBERT_emotion_val = tokenized_dataset_emotion_ALBERT['validation']


#_______________________________________________________________________________________________________________________________________________________________________________
# TASK 2: MODEL TRAINING AND EVALUATION
#_______________________________________________________________________________________________________________________________________________________________________________

num_labels=6
Albert_emotion_model=AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=num_labels)


# setting up the calculated hyperparameters
training_args = TrainingArguments(output_dir="./results", # Directory for saving outputs
                                  learning_rate=1.4002959791728854e-05, # Learning rate for optimization
                                  per_device_train_batch_size=8, # Batch size for training
                                  per_device_eval_batch_size=16, # Batch size for evaluation
                                  num_train_epochs=2, # Number of training epochs
                                  weight_decay=0.01, # Weight decay for regularization
                                  seed=4, # Seed that will be set at the beginning of training
                                  evaluation_strategy="epoch") # Evaluation is done at the end of each epoch
trainer_ALBERT_emotion = Trainer(model=Albert_emotion_model, args=training_args, train_dataset=ALBERT_emotion_train, eval_dataset=ALBERT_emotion_val, compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))})

eval_results_untrained=trainer_ALBERT_emotion.evaluate(ALBERT_emotion_test)
print(eval_results_untrained)
# fine-tuning the model
trainer_ALBERT_emotion.train()

# Saving the model
trainer_ALBERT_emotion.save_model("C:/Users/sangm/Onedrive/Desktop/4P84 trial/ALBERT_emotion")
tokenizer_ALBERT.save_pretrained("C:/Users/sangm/Onedrive/Desktop/4P84 trial/ALBERT_emotion")

# Evaluating the model
eval_results_trained = trainer_ALBERT_emotion.evaluate(ALBERT_emotion_test)
print(eval_results_trained)






