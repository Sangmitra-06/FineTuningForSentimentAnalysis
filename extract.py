import torch
from transformers import TrainingArguments

# Load the training args bin file
training_args = torch.load('DistilBERT_tweet/training_args.bin')
print(training_args)