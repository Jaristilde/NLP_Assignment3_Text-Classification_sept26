#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# clean_text_classification_template.py
"""
NLP Text Classification Template
-------------------------------
A clean, simplified template for binary text classification using NLP.

This template follows a step-by-step approach to transform raw text data
into a binary classification model using logistic regression.

Created for: Miami Dade College - AI Course
Date: September 19, 2025
"""

# ============================================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ============================================================================
# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Text processing libraries
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Download NLTK resources (uncomment if needed)
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')


# ============================================================================
# STEP 2: LOAD AND EXPLORE YOUR DATASET
# ============================================================================
# Load your dataset
data_path = "data/spam.csv"  # Path to the spam dataset in the data folder
df = pd.read_csv(data_path, encoding='latin-1')

# Display basic information
print("Dataset Information:")
print(f"Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())
print("\nColumn Names:")
print(df.columns.tolist())

# ============================================================================
# STEP 3: DATA CLEANING AND PREPARATION
# ============================================================================
# Identify text and target columns
text_column = "v2"    # The message text column
target_column = "v1"  # The label column (ham/spam)

# Check class distribution
print("\nClass distribution:")
print(df[target_column].value_counts())
print(df[target_column].value_counts(normalize=True).map(lambda x: f"{x:.2%}"))

# Remove rows with missing values
df = df.dropna(subset=[text_column, target_column])


# ============================================================================
# STEP 4: TEXT PREPROCESSING
# ============================================================================
# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize, remove stopwords, and lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # Join words back into a string
    return ' '.join(words)

# Apply preprocessing
df['processed_text'] = df[text_column].apply(preprocess_text)

# Show a sample of original vs processed text
print("\nOriginal vs Processed Text:")
for i in range(2):
    print(f"Original: {df[text_column].iloc[i][:100]}...")
    print(f"Processed: {df['processed_text'].iloc[i][:100]}...")
    print()


# ============================================================================
# STEP 5: SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================================
# Extract features and target
X = df['processed_text']
y = df[target_column]

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")


# ============================================================================
# STEP 6: FEATURE ENGINEERING WITH TF-IDF
# ============================================================================
# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=5000,
    min_df=5,
    max_df=0.8,
    stop_words='english'
)

# Transform text to TF-IDF features
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"Number of features: {X_train_tfidf.shape[1]}")


# ============================================================================
# STEP 7: TRAIN LOGISTIC REGRESSION MODEL
# ============================================================================
# Initialize and train the model
model = LogisticRegression(
    C=1.0,
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

# Train the model
model.fit(X_train_tfidf, y_train)


# ============================================================================
# STEP 8: EVALUATE MODEL PERFORMANCE
# ============================================================================
# Make predictions
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f"\nAccuracy: {accuracy:.4f}")

# Generate classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)


# ============================================================================
# STEP 9: VISUALIZE CONFUSION MATRIX
# ============================================================================
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],  # Class labels for predictions
            yticklabels=['Ham', 'Spam'])  # Class labels for actual values
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()


# ============================================================================
# STEP 10: ANALYZE FEATURE IMPORTANCE
# ============================================================================
# Get feature names and coefficients
feature_names = tfidf.get_feature_names_out()
coefficients = model.coef_[0]

# Create DataFrame of feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': coefficients
})

# Sort by absolute importance
feature_importance['Abs_Importance'] = abs(feature_importance['Importance'])
feature_importance = feature_importance.sort_values('Abs_Importance', ascending=False)

# Display top positive and negative features
print("\nTop 10 Positive Features:")
print(feature_importance[feature_importance['Importance'] > 0].head(10))

print("\nTop 10 Negative Features:")
print(feature_importance[feature_importance['Importance'] < 0].head(10))


# ============================================================================
# STEP 11: MAKE PREDICTIONS ON NEW TEXT
# ============================================================================
def predict_text(text):
    """Predict the class of new text"""
    # Preprocess the text
    processed = preprocess_text(text)
    
    # Transform using the same vectorizer
    text_tfidf = tfidf.transform([processed])
    
    # Make prediction
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0]
    
    return prediction, probability

# Example usage - let's test some messages
print("\nTesting the model with example messages:")

# Test 1: A typical ham message
new_text = "Hi, can we meet tomorrow at 3pm for coffee?"
prediction, probability = predict_text(new_text)
print("\nMessage 1:", new_text)
print(f"Prediction: {prediction}")
print(f"Confidence: {max(probability):.2%}")

# Test 2: A typical spam message
new_text = "CONGRATULATIONS! You've won a free iPhone! Click here to claim your prize now!"
prediction, probability = predict_text(new_text)
print("\nMessage 2:", new_text)
print(f"Prediction: {prediction}")
print(f"Confidence: {max(probability):.2%}")

# Test 3: Another ham message
new_text = "Don't forget to bring your laptop to the study session"
prediction, probability = predict_text(new_text)
print("\nMessage 3:", new_text)
print(f"Prediction: {prediction}")
print(f"Confidence: {max(probability):.2%}")


# ============================================================================
# STEP 12: SAVE MODEL FOR FUTURE USE
# ============================================================================
import pickle

# Save model and vectorizer
with open('text_classification_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("\nModel and vectorizer saved successfully.")
print("Text classification pipeline completed!")
