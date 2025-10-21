# SMS Spam Detection Using Machine Learning

This project implements a machine learning pipeline to classify SMS messages as either "spam" or "ham" (not spam). The system uses Natural Language Processing (NLP) techniques and a Logistic Regression model to achieve high accuracy.

## Project Overview

- **Problem:** Classify SMS text messages into spam or ham categories.
- **Solution:** A supervised learning model trained on the UCI SMS Spam Collection Dataset.
- **Technology:** Python, Scikit-learn, Pandas, NLTK.

## Project Report

For a detailed, formal description of the project, including methodology, architecture, and results, please see the full project document: [project_document.html](project_document.html)

## Repository Structure

```
sms-spam-detection/
├── data/                   # Dataset
├── models/                 # Trained model and vectorizer
├── src/                    # Source code
├── project_document.html   # Full project report
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/sms-spam-detection.git
    cd sms-spam-detection
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venvScriptsactivate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK data:**
    Run Python and enter the following:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    ```

## How to Run

### Training the Model

If you want to retrain the model from scratch, run the training script. The trained model and vectorizer will be saved in the `models/` directory.

```bash
python src/train.py
```

### Making Predictions

Use the prediction script to classify new SMS messages using the pre-trained model.

```bash
python src/predict.py "Congratulations! You've won a $1000 gift card."
```

**Output:**
```
Message: "Congratulations! You've won a $1000 gift card."
Prediction: spam
Confidence: 98.5%
```

## Dataset

The dataset used is the "SMS Spam Collection Data Set" from the UCI Machine Learning Repository. It is included in this repository in the `data/` folder for convenience.

- **Source:** [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)

## Model Performance

The final Logistic Regression model achieved the following performance on the test set:
- **Accuracy:** ~99%
- **Precision (Spam):** ~97%
- **Recall (Spam):** ~92%
- **F1-Score (Spam):** ~94%

## Future Enhancements

- [ ] Implement deep learning models (e.g., BERT).
- [ ] Add support for multiple languages.
- [ ] Deploy as a REST API.
