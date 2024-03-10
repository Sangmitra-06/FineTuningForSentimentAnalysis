from transformers import AlbertTokenizer, AlbertForSequenceClassification, \
    Trainer, TrainingArguments

# Loading your saved model
model_path = "Albert_tweet"
model = AlbertForSequenceClassification.from_pretrained(model_path,num_labels=2)

# Loading the tokenizer
tokenizer = AlbertTokenizer.from_pretrained(model_path)


# Pushing the model to the Hugging Face Model HubBERT
model.push_to_hub("Albert_tweet", use_auth_token="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")
