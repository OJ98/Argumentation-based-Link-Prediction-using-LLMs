import pandas as pd
from transformers import BertTokenizer, BertModel
import itertools
from feature_extraction import train_and_save_models, predict_quality, predict_argumentativeness, predict_sentiment


# Compute BERT embeddings for all tweets
def compute_bert_embeddings(tweet):
    # Load BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    # Tokenize the tweet
    inputs = tokenizer(tweet, return_tensors='pt', truncation=True, padding=True, max_length=512)
    # Get the hidden states from the model
    outputs = model(**inputs)
    # Extract the [CLS] token embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().detach().numpy()
    return cls_embedding

# Compute average BERT embedding for a given user
def aggregate_user_embeddings(userID, raw_tweets_df):
    # Filter the raw_tweets_df for the given user
    user_df = raw_tweets_df[raw_tweets_df['UserID'] == userID]
    # Compute the average BERT embedding for the user
    user_embedding = user_df['Embedding'].mean()
    return user_embedding

# Compute average Argument Quality for a given user
def aggregate_user_quality(userID, raw_tweets_df):
    # Filter the raw_tweets_df for the given user
    user_df = raw_tweets_df[raw_tweets_df['UserID'] == userID]
    # Compute the average argument quality for the user
    user_quality = user_df['Quality'].mean()
    return user_quality

# Compute average Argumentativeness for a given user
def aggregate_user_argumentativeness(userID, raw_tweets_df):
    # Filter the raw_tweets_df for the given user
    user_df = raw_tweets_df[raw_tweets_df['UserID'] == userID]
    # Compute the average argumentativeness for the user
    user_argumentativeness = user_df['Argumentativeness'].mean()
    return user_argumentativeness

# Compute average Sentiment for a given user
def aggregate_user_sentiment(userID, raw_tweets_df):
    # Filter the raw_tweets_df for the given user
    user_df = raw_tweets_df[raw_tweets_df['UserID'] == userID]
    # Compute the average sentiment for the user
    user_sentiment = user_df['Sentiment'].mean()
    return user_sentiment

# Return 1 if relationship exists between two users and 0 otherwise
def read_ground_truth(user1, user2, ground_truth_df):
    # Check if the relationship exists in the ground_truth_df
    if ((ground_truth_df['User1'] == user1) & (ground_truth_df['User2'] == user2)).any():
        return 1
    elif ((ground_truth_df['User1'] == user2) & (ground_truth_df['User2'] == user1)).any():
        return 1
    else:
        return 0

# Read raw_tweets.csv
raw_tweets_df = pd.read_csv('raw_tweets.csv')

# Compute BERT embeddings for all tweets
raw_tweets_df['Embedding'] = raw_tweets_df['Tweet'].apply(compute_bert_embeddings)

# Train and save models for argument feature extraction
train_and_save_models()

# Predict the argument quality for each tweet
raw_tweets_df['Quality'] = raw_tweets_df['Tweet'].apply(predict_quality)

# Predict the argumentativeness of each tweet
raw_tweets_df['Argumentativeness'] = raw_tweets_df['Tweet'].apply(predict_argumentativeness)

# Predict the sentiment of each tweet
raw_tweets_df['Sentiment'] = raw_tweets_df['Tweet'].apply(predict_sentiment)

# Extract the unique users in the sampled dataset
unique_users = raw_tweets_df.iloc[:, 1].unique()

# Compute user embeddings for all users and store in a dictionary
user_embeddings = {}
for user in unique_users:
    user_embeddings[user] = aggregate_user_embeddings(user, raw_tweets_df)

# Load the ground truth dataset
ground_truth_df = pd.read_csv('user_pairs.csv')

# Generate all unique pairs of users
user_pairs = list(itertools.combinations(unique_users, 2))

# Convert the list of pairs to a DataFrame
final_df = pd.DataFrame(user_pairs, columns=['User1', 'User2'])

# Add a new column to final_df to store the extracted features and the relationship between users
final_df['User1_Embedding'] = None
final_df['User2_Embedding'] = None
final_df['User1_Quality']  = None
final_df['User2_Quality']  = None
final_df['User1_Argumentativeness']  = None
final_df['User2_Argumentativeness']  = None
final_df['User1_Sentiment']  = None
final_df['User2_Sentiment']  = None
final_df['relationship'] = 0

# Iterate over all rows in final_df
for index, row in final_df.iterrows():
    # Extract the user embeddings for both users in the pair
    user1_embedding = user_embeddings[row['User1']]
    user2_embedding = user_embeddings[row['User2']]
    # Extract the user quality for both users in the pair
    user1_quality = aggregate_user_quality(row['User1'], raw_tweets_df)
    user2_quality = aggregate_user_quality(row['User2'], raw_tweets_df)
    # Extract the user argumentativeness for both users in the pair
    user1_argumentativeness = aggregate_user_argumentativeness(row['User1'], raw_tweets_df)
    user2_argumentativeness = aggregate_user_argumentativeness(row['User2'], raw_tweets_df)
    # Extract the user sentiment for both users in the pair
    user1_sentiment = aggregate_user_sentiment(row['User1'], raw_tweets_df)
    user2_sentiment = aggregate_user_sentiment(row['User2'], raw_tweets_df)
    # Extract the relationship between the users in the pair
    relationship = read_ground_truth(row['User1'], row['User2'], ground_truth_df)
    # Store the features in the final_df
    final_df.at[index, 'User1_Embedding'] = user1_embedding
    final_df.at[index, 'User1_Quality'] = user1_quality
    final_df.at[index, 'User1_Argumentativeness'] = user1_argumentativeness
    final_df.at[index, 'User1_Sentiment'] = user1_sentiment
    final_df.at[index, 'User2_Embedding'] = user2_embedding
    final_df.at[index, 'User2_Quality'] = user2_quality
    final_df.at[index, 'User2_Argumentativeness'] = user2_argumentativeness
    final_df.at[index, 'User2_Sentiment'] = user2_sentiment
    final_df.at[index, 'relationship'] = relationship

print(final_df.head())
# Save the final_df to a CSV file
final_df.to_csv('preprocessed_dataset.csv', index=False)