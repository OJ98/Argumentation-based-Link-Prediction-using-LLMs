from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from canary.argument_pipeline import download_model, load_model
import textblob
from datasets import load_dataset
import torch

def train_and_save_models():
    # Download the argument detector models
    download_model("all")

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    ds = load_dataset("ibm/argument_quality_ranking_30k", "argument_quality_ranking")

    # Define the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    tokenized_datasets = ds.map(tokenize_function, batched=True)

    # Prepare the dataset for training
    train_dataset = tokenized_datasets['train']
    eval_dataset = tokenized_datasets['test']

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Train the model
    trainer.train()

    # Save the trained model
    model.save_pretrained('./trained_arg_quality_model')
    return model.name_or_path

# Predicts the argument quality of a given text from 0 to 1
def predict_quality(text):
    # Load the trained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('./trained_arg_quality_model')
    model = BertForSequenceClassification.from_pretrained('./trained_arg_quality_model')
    # Check if GPU is available and move model to GPU if possible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    score = torch.sigmoid(logits).item()  # Score between 0 and 1 for argument quality
    return score

# Predicts the argumentativeness of a given text from 0 to 1
def predict_argumentativeness(text):
    # load the detector
    detector = load_model("argument_detector")
    return detector.predict(text)

# Predicts the polarity of a given text from -1 to 1
def predict_sentiment(text):
    return textblob.TextBlob(text).sentiment.polarity

if __name__ == "__main__":
    train_and_save_models()