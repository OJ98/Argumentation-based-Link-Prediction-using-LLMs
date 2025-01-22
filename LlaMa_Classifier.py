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
    def format_prompt(self, user1_tweets, user2_tweets):
        prompt = (
                "Analyze the relationship between two users based on their tweets:\n\n"
                f"User 1 tweets: {', '.join(user1_tweets)}\n"
                f"User 2 tweets: {', '.join(user2_tweets)}\n"
                "Do these users have a significant relationship? "
                "Respond with a clear Yes or No."
        )
        return prompt
    
    # Zero-shot prediction of relationship between two users
    def predict(self, tweet_set1, tweet_set2):
        # Prepare the prompt
        prompt = self.format_prompt(tweet_set1, tweet_set2)
        
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
    
    def evaluate_dataset(self, tweet_sets1, tweet_sets2, true_labels):
        predictions = []
        reasoning_logs = []
        
        # Predict for each sample
        for ts1, ts2, feat, true_label in zip(tweet_sets1, tweet_sets2, true_labels):
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
    # Initialize the classifier
    classifier = LlamaTweetClassifier()
    
    # Evaluate the model
    results = classifier.evaluate_dataset(tweet_sets1, tweet_sets2, true_labels)
    
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