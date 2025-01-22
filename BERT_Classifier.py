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
    def __init__(self, embeddings1, embeddings2, labels):
        self.embeddings1 = embeddings1
        self.embeddings2 = embeddings2
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding1 = torch.tensor(self.User1_Embedding[idx], dtype=torch.float)
        embedding2 = torch.tensor(self.User2_Embedding[idx], dtype=torch.float)
        # Concatenate the two embeddings
        embedding = torch.cat((embedding1, embedding2), dim=0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return embedding, label

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create the custom datasets
train_dataset = CustomDataset(train_df['User1_Embedding'].tolist(), train_df['User2_Embedding'].tolist(), train_df['relationship_label'].tolist())
test_dataset = CustomDataset(test_df['User1_Embedding'].tolist(), test_df['User2_Embedding'].tolist(), test_df['relationship_label'].tolist())

# Define the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

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
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained('./BERT_trained_model')

# Evaluate the model
predictions = trainer.predict(test_dataset)
preds = torch.argmax(predictions.predictions, axis=1)
probs = torch.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1]

# Compute the classification report
report = classification_report(test_df['relationship'], preds, target_names=['Class 0', 'Class 1'])
print(report)

# Compute the AUC
auc = roc_auc_score(test_df['relationship'], probs)
print(f"AUC: {auc}")