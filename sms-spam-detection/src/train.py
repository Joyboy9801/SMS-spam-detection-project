import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# --- Configuration ---
DATA_PATH = '../data/spam.csv'
MODEL_DIR = '../models'
MODEL_FILE = os.path.join(MODEL_DIR, 'logistic_regression_model.joblib')
VECTORIZER_FILE = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')

# --- Ensure NLTK data is downloaded ---
try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt...")
    nltk.download('punkt')

# --- 1. Data Loading and Preprocessing ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

print("Loading and preprocessing data...")
df = pd.read_csv(DATA_PATH, sep='\t', header=None, names=['label', 'message'])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df['processed_message'] = df['message'].apply(preprocess_text)

# --- 2. Feature Extraction ---
print("Extracting features with TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(max_features=3000)
X = tfidf_vectorizer.fit_transform(df['processed_message'])
y = df['label']

# --- 3. Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 4. Model Training ---
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- 5. Evaluation ---
print("\nEvaluating model...")
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- 6. Save Model and Vectorizer ---
print(f"\nSaving model to {MODEL_FILE} and vectorizer to {VECTORIZER_FILE}...")
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_FILE)
joblib.dump(tfidf_vectorizer, VECTORIZER_FILE)

print("Training complete!")
