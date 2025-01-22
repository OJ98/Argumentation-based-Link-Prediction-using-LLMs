import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Extract the tweet sets for each user
def extract_tweet_sets(raw_tweets, UserID):
    # Filter the raw_tweets for the given UserID
    user_tweets = raw_tweets[raw_tweets['UserID'] == UserID]
    # Extract the tweet sets for the user
    tweet_set = user_tweets['Tweet'].values
    return tweet_set

class LlamaTweetClassifier:
    def __init__(self, model_path='meta-llama/Llama-3.1-8B-Instruct'):
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # Use half precision for memory efficiency 
            torch_dtype=torch.float16,  
            # Automatically use available GPU/CPU
            device_map='auto'  
        )
        
        # Configure generation parameters
        self.generation_config = {
            # Limit response length
            'max_new_tokens': 10,
            # Enable sampling
            'do_sample': True,
            # Lower temperature for more deterministic responses
            'temperature': 0.7,
            # Nucleus sampling
            'top_p': 0.9,  
        }
    
    # Returns formatted prompt
    def format_prompt(self, user1_tweets, user2_tweets, features):
        prompt =  (
                "Analyze the relationship between two users based on their tweets:\n\n"
                f"User 1 tweets: {', '.join(user1_tweets)}\n"
                "Argumentation Quality Score for User 1: {:.2f}\n"
                "Argumentativeness Score for User 1: {:d}\n"
                "Argumentation Semantic Score for User 1: {:.2f}\n\n"
                f"User 2 tweets: {', '.join(user2_tweets)}\n"
                "Argumentation Quality Score for User 2: {:.2f}\n"
                "Argumentativeness Score for User 2: {:d}\n"
                "Argumentation Semantic Score for User 2: {:.2f}\n"
                "Do these users have a significant relationship? "
                "Respond with a clear Yes or No."
            ).format(features['U1_Quality'], features['U1_Arg'], features['U1_Sem'],features['U2_Quality'], features['U2_Arg'], features['U2_Sem'])

        return prompt
    
    # Zero-shot prediction of relationship between two users
    def predict(self, tweet_set1, tweet_set2, features):
        # Prepare the prompt
        prompt = self.format_prompt(tweet_set1, tweet_set2, features)
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate response
        outputs = self.model.generate(**inputs, **self.generation_config)
        
        # Decode the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the prediction and reasoning
        try:
            # Try to extract the label from the response
            prediction_text = response.split("Predicted Label:")[-1].strip()
            
            # Extract the first numeric digit (0 or 1)
            predicted_label = int(''.join(filter(str.isdigit, prediction_text))[0])
            
            return predicted_label, response
        except:
            # Fallback to random prediction if parsing fails
            return np.random.randint(2), response
    
    def evaluate_dataset(self, tweet_sets1, tweet_sets2, features, true_labels):
        predictions = []
        reasoning_logs = []
        
        # Predict for each sample
        for ts1, ts2, feat, true_label in zip(tweet_sets1, tweet_sets2, features, true_labels):
            pred, reasoning = self.predict(ts1, ts2, feat)
            predictions.append(pred)
            reasoning_logs.append(reasoning)
        
        # Calculate metrics
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        accuracy = np.mean(predictions == true_labels)
        precision = np.sum((predictions == 1) & (true_labels == 1)) / max(np.sum(predictions == 1), 1)
        recall = np.sum((predictions == 1) & (true_labels == 1)) / max(np.sum(true_labels == 1), 1)
        f1 = 2 * (precision * recall) / max((precision + recall), 1)
        auc = roc_auc_score(true_labels, predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': auc,
            'predictions': predictions,
            'reasoning_logs': reasoning_logs
        }

def main():
    # Load the raw tweet dataset
    raw_tweets = pd.read_csv('raw_tweets.csv')
    # Load the feature dataset
    feature_set = pd.read_csv('preprocessed_dataset.csv')
    print(feature_set.head())
    # Extract ground truth
    true_labels = feature_set['relationship'].values
    # Extract tweet sets
    tweet_sets1 = [extract_tweet_sets(raw_tweets, uid) for uid in feature_set['UserID1']]
    tweet_sets2 = [extract_tweet_sets(raw_tweets, uid) for uid in feature_set['UserID2']]
    # Extract features
    U1_Quality = feature_set['User1_Quality'].values,
    U2_Quality = feature_set['User2_Quality'].values,
    U1_Arg = feature_set['User1_Argumentativeness'].values,
    U2_Arg = feature_set['User2_Argumentativeness'].values,
    U1_Sem = feature_set['User1_Sentiment'].values,
    U2_Sem =  feature_set['User2_Sentiment'].values,
    # Construct feature dictionary
    features = []
    for i in range(len(U1_Quality)):
        features.append({
            'U1_Quality': U1_Quality[i],
            'U2_Quality': U2_Quality[i],
            'U1_Arg': U1_Arg[i],
            'U2_Arg': U2_Arg[i],
            'U1_Sem': U1_Sem[i],
            'U2_Sem': U2_Sem[i]
        })
    # Initialize the classifier
    classifier = LlamaTweetClassifier()
    
    # Evaluate the model
    results = classifier.evaluate_dataset(tweet_sets1, tweet_sets2, features, true_labels)
    
    # Print results
    print("Classification Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"ROC AUC: {results['roc_auc']:.4f}")
    
    # Print individual predictions and reasoning logs (for debugging)
    # for i, (pred, true, reasoning) in enumerate(zip(results['predictions'], true_labels, results['reasoning_logs'])):
    #     print(f"\nSample {i+1}:")
    #     print(f"Predicted: {pred}, True Label: {true}")
    #     print("Model Reasoning:")
    #     print(reasoning)

if __name__ == "__main__":
    main()