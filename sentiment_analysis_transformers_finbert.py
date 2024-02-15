import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import click

# Load the pre-trained model
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def get_key_with_max_value(data_dict):
    # Find the key with the maximum value
    max_key = max(data_dict, key=data_dict.get)
    return max_key

def analyze_sentiment(news):
        
    tokenizer_kwargs = {"padding": True, "truncation": True}
    input_sequence = tokenizer(news, return_tensors="pt", **tokenizer_kwargs)
    
    outputs = model(**input_sequence)
    
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    positive = predictions[:, 0].tolist()[0]
    negative = predictions[:, 1].tolist()[0]
    neutral = predictions[:, 2].tolist()[0]
    
    
    predicted_sentiment = {"positive": positive, "negative": negative, "neutral": neutral}
    
    max_key = get_key_with_max_value(predicted_sentiment)
    
    print("Sentiment: " + str(max_key) + " with score: " + str(predicted_sentiment[max_key]))
    print("Positive: " + str(positive))
    print("Negative: " + str(negative))
    print("Neutral: " + str(neutral))

@click.command()
@click.argument('input_string')
def sentiment_analysis_result(input_string):
    
    analyze_sentiment(input_string)
    

if __name__ == '__main__':
    
    sentiment_analysis_result()
    