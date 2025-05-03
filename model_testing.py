from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

# Load the saved model and tokenizer
model_path = "./model"  # Path to your saved model
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
emotion_list = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval",
    "disgust","embarrassment","excitement","fear","gratitude","grief",
    "joy","love","nervousness","optimism","pride","realization",
    "relief","remorse","sadness","surprise","neutral",
]

# Function to predict emotions for a given text
def predict_emotions(text):
    # Tokenize the input text
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Get predicted probabilities
    probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]  
    
    # Get predicted emotions based on a threshold (e.g., 0.5)
    predicted_emotions = [emotion_list[i] for i, prob in enumerate(probs) if prob > 0.5]  
    dic = []
    for i in range(len(probs)):
        dic.append((probs[i], emotion_list[i]))
    # print(dic)
    dic.sort(reverse=True)
    predicted_emotions = [x[1] for x in dic[0:3]]
    # emotion_list is from cell 11 in your original code.
    return predicted_emotions

# Example usage
text = ""  # Your input text
predicted_emotions = predict_emotions(text)
print(f"Predicted Emotions: {predicted_emotions}")