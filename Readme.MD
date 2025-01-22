# Argumentation based Link Prediction using LLMs

## Dependancies
```pandas```\
```canary```\
```textblob```\
```transformers```\
```pytorch```\
```datasets```\
```llama```

run ```pip install git+https://github.com/chriswales95/Canary.git@development``` to install ```canary```

## Input Data

```raw_tweets.csv``` contains the shortlisted 10K tweets
```user_pairs.csv``` contains the user relationships between the users in ```raw_tweets.csv```


## Execution Instructions

Step 1: Run ```python preprocess.py``` - It will take a long time as it computes the BERT embeddings and ArgFeatures for each tweet and then aggregates it across all tweets for a single user. The output is written to ```preprocessed_dataset.csv```

Step 2: Run ```python BERT_Classifier.py``` - It will train the BERT model on the embeddings only and output classification results.

Step 3: Run ```python BERT_Classifier_ArgFeatures.py``` - It will train the BERT model on the embeddings and ArgFeatures and output classification results.

Step 4: Run ```python LlaMa_Classifier.py``` - It will train an instance of LlaMa 3.1-8B on the tweet set only and output classification results.

Step 5: Run ```python LlaMa_Classifier_ArgFeatures.py``` - It will train an instance of LlaMa 3.1-8B on the tweet set and ArgFeatures and output classification results.

Step 6: Run ```python ablation_study.py``` - It will run the ablation study experiment and output the feature relevance table. Since this trains three additional models and saves them, it takes a long time to run.