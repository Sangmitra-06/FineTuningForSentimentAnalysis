'''
    Manual hyperparameter search for DistilBERT
'''
from datasets import load_dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments, \
    EarlyStoppingCallback
import numpy as np
from sklearn.metrics import accuracy_score

# function to tokenize dataset
def tokenize_function(example, tokenizer):
    return tokenizer(example["text"], padding="max_length", truncation=True)


# Loading the dataset
dataset = load_dataset('tweet_eval', 'irony')

# Tokenizing the dataset
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokenized_dataset = dataset.map(lambda example: tokenize_function(example, tokenizer), batched=True)

# Spliting the dataset into train, test, and validation sets
train_dataset = tokenized_dataset['train']
test_dataset = tokenized_dataset['test']
val_dataset = tokenized_dataset['validation']


# Defining the model initialization function
def model_init():
    return DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Defining the hyperparameter search space
hyperparameter_space = {
    "learning_rate": [5e-05, 4e-05],
    "num_train_epochs": [3, 4, 5],
    "per_device_train_batch_size": [10, 16, 32, 64]
}

# Defining the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    logging_dir="./logs",
    per_device_eval_batch_size=16,
    logging_steps=10,
    remove_unused_columns=False,
    disable_tqdm=False,
    seed=42,
    load_best_model_at_end=True
)
# Defining the Trainer
trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))}, callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
# Performing an iterative hyperparameter search
best_accuracy = 0
best_hyperparameters = None

for lr in hyperparameter_space["learning_rate"]:
    for epochs in hyperparameter_space["num_train_epochs"]:
        for batch_size in hyperparameter_space["per_device_train_batch_size"]:
            training_args.learning_rate = lr
            training_args.num_train_epochs = epochs
            training_args.per_device_train_batch_size = batch_size

            trainer.args = training_args
            trainer.train()

            eval_result = trainer.evaluate(eval_dataset=test_dataset)
            accuracy = eval_result["eval_accuracy"]

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_hyperparameters = {
                    "learning_rate": lr,
                    "num_train_epochs": epochs,
                    "per_device_train_batch_size": batch_size
                }
            print("Accuracy best so far", best_accuracy)
            print(best_hyperparameters)

# Printing the best hyperparameters and accuracy
print("Best hyperparameters:", best_hyperparameters)
print("Best accuracy:", best_accuracy)
