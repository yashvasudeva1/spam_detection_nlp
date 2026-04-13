# Spam vs Ham Detection System  
End-to-End NLP Project with Imbalanced Data Handling and Streamlit Deployment

---

## Project Overview

This project is an end-to-end Natural Language Processing (NLP) based spam detection system designed to classify SMS messages as Spam or Ham (Not Spam).

Unlike basic text classification projects, this system focuses on real-world machine learning challenges such as:

- Handling highly imbalanced datasets  
- Choosing appropriate evaluation metrics beyond accuracy  
- Decision threshold tuning to manage business trade-offs  
- Model explainability at the word level  
- Production-aware deployment using Streamlit  

The final solution is deployed as an interactive, inference-only Streamlit web application.

---

## Problem Statement

Spam messages occur far less frequently than legitimate messages, making spam detection an imbalanced classification problem.

Optimizing only for accuracy often:
- Misses spam messages (high false negatives)
- Appears to perform well numerically but fails in real-world usage

This project addresses the issue by prioritizing spam recall and explicitly managing the precision–recall trade-off.

---

## Project Classification

- Primary Domain: Applied Machine Learning  
- Secondary Domains:
  - Natural Language Processing (NLP)
  - Imbalanced Classification
  - Model Explainability
  - Introductory MLOps and Deployment  

Project Level: Strong Intermediate (Borderline Advanced for undergraduate level)

---

## Dataset

- Dataset: SMS Spam Collection Dataset  
- Total Messages: 5,572  
- Class Distribution:
  - Ham: Approximately 87%
  - Spam: Approximately 13%

The imbalance is intentional and reflects real-world spam detection systems.

---

## Technology Stack

### Core Libraries
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- NLTK  

### NLP and Modeling
- TF-IDF Vectorization  
- Multinomial Naive Bayes  
- Threshold-based classification  

### Deployment and Visualization
- Streamlit  
- Plotly  

---

## Project Workflow

Raw SMS Text
↓
Text Cleaning and Lemmatization
↓
Train–Test Split (Stratified)
↓
TF-IDF Vectorization (fit on training data only)
↓
Multinomial Naive Bayes Training
↓
Threshold Tuning for Class Imbalance
↓
Evaluation using Recall, Precision, and F1-score
↓
Pipeline Serialization
↓
Streamlit Deployment (Inference Only)

yaml
Copy code

---

## Text Preprocessing

The following preprocessing steps are applied:

- Removal of non-alphabetic characters  
- Lowercasing  
- Tokenization  
- Stopword removal  
- Lemmatization  

No statistical preprocessing steps (such as TF-IDF fitting) are performed before the train–test split to prevent data leakage.

---

## Model Selection

Multinomial Naive Bayes was chosen because:

- It performs well on sparse, high-dimensional text data  
- It is computationally efficient  
- It provides probabilistic outputs useful for threshold tuning  
- It offers interpretability through word-level probabilities  

The model and TF-IDF vectorizer are combined using a Scikit-learn Pipeline for clean deployment.

---

## Handling Class Imbalance

### Observations
- The dataset is heavily skewed toward Ham messages  
- Default classification thresholds bias predictions toward Ham  

### Solutions Implemented
- Decision threshold tuning  
- Evaluation using recall, precision, and F1-score instead of accuracy alone  
- Acceptance of reduced precision to significantly improve spam recall  

This mirrors real-world spam filtering systems, where false negatives are more costly than false positives.

---

## Model Performance (After Threshold Tuning)

| Class | Precision | Recall | F1-Score |
|------|----------|--------|---------|
| Ham | 0.995 | 0.913 | 0.953 |
| Spam | 0.633 | 0.973 | 0.767 |

**Key Insight:**  
The model successfully detects more than 97% of spam messages, minimizing spam leakage.

---

## Model Explainability

Since Multinomial Naive Bayes is a probabilistic model, word-level explainability is incorporated by:

- Identifying top spam-indicative words  
- Identifying top ham-indicative words  
- Visualizing word importance using ranked charts  

This provides transparency into how the model makes predictions.

---

## Streamlit Application Features

### Application Pages
- Home: Project overview and dataset summary  
- Data Overview: Class distribution and message statistics  
- Model Performance: Metrics, confusion matrix, classification report  
- Make Predictions:
  - User-entered message classification  
  - Spam probability visualization  
  - Threshold-aware decision logic  
- Model Insights:
  - Top spam and ham indicators  
  - Explainability visualizations  

### Deployment Principles
- No model retraining inside Streamlit  
- Inference-only usage  
- Serialized pipeline loading  
- Clean, analytics-focused user interface  

---

## Project Workflow

```
spam-detection-nlp/
│
├── app.py                  # Streamlit application (inference only)
├── train_model.py          # Model training and pipeline serialization
├── spam_model.pkl          # Saved TF-IDF + Naive Bayes pipeline
├── spam.csv                # SMS Spam Collection dataset
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
└── .gitignore              # Git ignore file
```
