import joblib
import sys
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

# --- Configuration ---
MODEL_DIR = '../models'
MODEL_FILE = os.path.join(MODEL_DIR, 'logistic_regression_model.joblib')
VECTORIZER_FILE = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')

# --- Load Model and Vectorizer ---
try:
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
except FileNotFoundError:
    print("Error: Model or vectorizer not found.")
    print("Please run 'python src/train.py' first to train and save the model.")
    sys.exit(1)

# --- Preprocessing function (must be identical to the one in train.py) ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# --- Prediction Function ---
def predict_spam(message):
    processed_message = preprocess_text(message)
    vectorized_message = vectorizer.transform([processed_message])
    prediction = model.predict(vectorized_message)[0]
    probability = model.predict_proba(vectorized_message)[0]
    
    label = 'spam' if prediction == 1 else 'ham'
    confidence = max(probability) * 100
    
    return label, confidence

# --- Main execution ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"Your SMS message here\"")
        sys.exit(1)
    
    input_message = " ".join(sys.argv[1:])
    
    label, confidence = predict_spam(input_message)
    
    print(f"\nMessage: \"{input_message}\"")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}%")
