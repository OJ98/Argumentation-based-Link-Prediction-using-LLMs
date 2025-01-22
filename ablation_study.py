import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score

# Load the preprocessed dataset
df = pd.read_csv('./preprocessed_dataset.csv')

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, embeddings1, embeddings2, features, labels):
        self.embeddings1 = embeddings1
        self.embeddings2 = embeddings2
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding1 = torch.tensor(self.User1_Embedding[idx], dtype=torch.float)
        embedding2 = torch.tensor(self.User2_Embedding[idx], dtype=torch.float)
        # Concatenate the two embeddings
        embedding = torch.cat((embedding1, embedding2), dim=0)
        # Get the additional features
        feature = torch.tensor(self.features[idx], dtype=torch.float)
        # Concatenate the embeddings with the additional features
        combined = torch.cat((embedding, feature), dim=0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return combined, label

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Extract Argumentation features stepwise
arg_features_quality = ['User1_Argumentativeness', 'User1_Sentiment', 'User2_Argumentativeness', 'User2_Sentiment']
arg_features_argumentativeness = ['User1_Quality', 'User1_Sentiment', 'User2_Quality', 'User2_Sentiment']
arg_features_sentiment = ['User1_Quality', 'User1_Argumentativeness', 'User2_Quality', 'User2_Argumentativeness']

# Create datasets for each set of features
# Quality 
train_dataset_quality = CustomDataset(
    train_df['User1_Embedding'].tolist(),
    train_df['User2_Embedding'].tolist(),
    train_df[arg_features_quality].values.tolist(),
    train_df['relationship'].tolist()
)
test_dataset_quality = CustomDataset(
    test_df['User1_Embedding'].tolist(),
    test_df['User2_Embedding'].tolist(),
    test_df[arg_features_quality].values.tolist(),
    test_df['relationship'].tolist()
)
# Argumentativeness
train_dataset_argumentativeness = CustomDataset(
    train_df['User1_Embedding'].tolist(),
    train_df['User2_Embedding'].tolist(),
    train_df[arg_features_argumentativeness].values.tolist(),
    train_df['relationship'].tolist()
)
test_dataset_argumentativeness = CustomDataset(
    test_df['User1_Embedding'].tolist(),
    test_df['User2_Embedding'].tolist(),
    test_df[arg_features_argumentativeness].values.tolist(),
    test_df['relationship'].tolist()
)
# Sentiment
train_dataset_sentiment = CustomDataset(
    train_df['User1_Embedding'].tolist(),
    train_df['User2_Embedding'].tolist(),
    train_df[arg_features_sentiment].values.tolist(),
    train_df['relationship'].tolist()
)
test_dataset_sentiment = CustomDataset(
    test_df['User1_Embedding'].tolist(),
    test_df['User2_Embedding'].tolist(),
    test_df[arg_features_sentiment].values.tolist(),
    test_df['relationship'].tolist()
)

# Define the models
model_quality = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model_argumentativeness = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model_sentiment = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

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

# Define the Trainers for each set of features
# Quality
trainer_quality = Trainer(
    model=model_quality,
    args=training_args,
    train_dataset=train_dataset_quality,
    eval_dataset=test_dataset_quality,
)
# Argumentativeness
trainer_argumentativeness = Trainer(
    model=model_argumentativeness,
    args=training_args,
    train_dataset=train_dataset_argumentativeness,
    eval_dataset=test_dataset_argumentativeness,
)
# Sentiment
trainer_sentiment = Trainer(
    model=model_sentiment,
    args=training_args,
    train_dataset=train_dataset_sentiment,
    eval_dataset=test_dataset_sentiment,
)

# Train the models
trainer_quality.train()
trainer_argumentativeness.train()
trainer_sentiment.train()

# Save the trained models
model_quality.save_pretrained('./BERT_ArgFeatures_NoQuality_trained_model')
model_argumentativeness.save_pretrained('./BERT_ArgFeatures_NoArgumentativeness_trained_model')
model_sentiment.save_pretrained('./BERT_ArgFeatures_NoSentiment_trained_model')

# Evaluate the models
# Quality
predictions_quality = trainer_quality.predict(test_dataset_quality)
preds_quality = torch.argmax(predictions_quality.predictions, axis=1)
probs_quality = torch.softmax(torch.tensor(predictions_quality.predictions), dim=1)[:, 1]
# Argumentativeness
predictions_argumentativeness = trainer_argumentativeness.predict(test_dataset_argumentativeness)
preds_argumentativeness = torch.argmax(predictions_argumentativeness.predictions, axis=1)
probs_argumentativeness = torch.softmax(torch.tensor(predictions_argumentativeness.predictions), dim=1)[:, 1]
# Sentiment
predictions_sentiment = trainer_sentiment.predict(test_dataset_sentiment)
preds_sentiment = torch.argmax(predictions_sentiment.predictions, axis=1)
probs_sentiment = torch.softmax(torch.tensor(predictions_sentiment.predictions), dim=1)[:, 1]

# Argument Quality Score
print("Argument Quality Results:")
# Compute the classification report
report_quality = classification_report(test_df['relationship'], preds_quality, target_names=['Class 0', 'Class 1'])
print(report_quality)
# Compute the AUC
auc = roc_auc_score(test_df['relationship'], probs_quality)
print(f"AUC: {auc}")

# Argumentativeness Score
print("Argumentativeness Results:")
# Compute the classification report
report_argumentativeness = classification_report(test_df['relationship'], preds_argumentativeness, target_names=['Class 0', 'Class 1'])
print(report_argumentativeness)
# Compute the AUC
auc = roc_auc_score(test_df['relationship'], probs_argumentativeness)
print(f"AUC: {auc}")

# Sentiment Score
print("Sentiment Results:")
# Compute the classification report
report_sentiment = classification_report(test_df['relationship'], preds_sentiment, target_names=['Class 0', 'Class 1'])
print(report_sentiment)
# Compute the AUC
auc = roc_auc_score(test_df['relationship'], probs_sentiment)
print(f"AUC: {auc}")