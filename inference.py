import torch 
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

MODEL_PATH = r"./training_runs/checkpoint-105" 
INPUT_FILE = r"./tweet_samples.txt" 
MAX_LENGTH = 128 # 
LABELS = { 0: "negative", 1: "neutral", 2: "positive" } #match labels with dataset labels

#  tokenizer initialise
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH) 
tokenizer.pad_token = tokenizer.eos_token # Set pad token to eos token

#loading model
model = GPT2ForSequenceClassification.from_pretrained(MODEL_PATH)
model.config.pad_token_id = tokenizer.pad_token_id

device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" ) 
print("Using device:", device) 
model.to(device) 
model.eval()

#Prediction
def predict_sentiment(text): 
    inputs = tokenizer( text, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt" ) 
    inputs = { key: value.to(device) for key, value in inputs.items() } 
    with torch.no_grad(): outputs = model(**inputs) 
    logits = outputs.logits 
    probabilities = torch.softmax(logits, dim=-1) 
    predicted_class = torch.argmax( probabilities, dim=-1 ).item() 
    confidence = probabilities[ 0, predicted_class ].item() 
    return ( 
        predicted_class, LABELS[predicted_class], 
        confidence, 
        probabilities[0].cpu().numpy() )

# For the test purpose, reading tweet/feedback from a text file

with open(INPUT_FILE, "r", encoding="utf-8") as file: 
    tweets = [ line.strip() for line in file if line.strip() ]

print(f"Loaded {len(tweets)} tweets")

# Run inference
for i, tweet in enumerate(tweets, start=1):
    predicted_class, label, confidence, probs = predict_sentiment(tweet)
    print(f"Tweet: {tweet}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Label: {label}")
    print(f"Confidence: {confidence}")
    print(f"Probabilities: {probs}")
    print("-" * 50)

    for class_id, probability in enumerate(probs): 
        print( f" {LABELS[class_id]}: " f"{probability:.4f}" )
