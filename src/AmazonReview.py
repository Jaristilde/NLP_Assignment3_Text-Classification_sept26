#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Large-Scale Amazon Reviews Sentiment Analysis
-------------------------------------------
This script performs sentiment analysis on a large dataset of Amazon product reviews,
processing the data in chunks to handle memory efficiently.

Dataset:
- 3.6M+ reviews
- Binary sentiment (positive/negative)
- Text reviews with __label__ prefix
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc  # Garbage collector for memory management
from tqdm import tqdm  # Progress bar
import bz2
import logging
import kagglehub  # For downloading the dataset

# Text processing libraries
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Download Amazon reviews dataset
print("Downloading Amazon reviews dataset...")
path = kagglehub.dataset_download("bittlingmayer/amazonreviews")
print("Path to dataset files:", path)

# Function to read and preprocess the data
def load_amazon_reviews(file_path, sample_size=None):
    """
    Load and preprocess Amazon reviews data with memory-efficient processing
    
    Parameters:
    file_path (str): Path to the dataset
    sample_size (int): Number of reviews to sample (None for all)
    
    Returns:
    pd.DataFrame: Processed DataFrame with reviews and labels
    """
    import bz2
    import random
    random.seed(42)
    
    # If sample size is specified, calculate sampling probability
    total_lines = 0
    with bz2.open(file_path, 'rt', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    
    sampling_prob = sample_size / total_lines if sample_size else 1.0
    
    reviews = []
    labels = []
    count = 0
    
    with bz2.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
                
            # Random sampling
            if random.random() > sampling_prob:
                continue
                
            # Extract label and review
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                label, review = parts
                labels.append(label)
                reviews.append(review)
                count += 1
                
            # Break if we have enough samples
            if sample_size and count >= sample_size:
                break
                
            # Free up memory periodically
            if count % 1000 == 0:
                import gc
                gc.collect()
    
    # Create DataFrame
    df = pd.DataFrame({
        'sentiment': labels,
        'review': reviews
    })
    
    # Convert sentiment to binary (1 for positive, 0 for negative)
    df['sentiment'] = (df['sentiment'] == '__label__2').astype(int)
    
    # Remove any rows with missing values
    df = df.dropna()
    
    return df

# Text preprocessing function
def preprocess_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize, remove stopwords, and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # Join words back into a string
    return ' '.join(words)

def main():
    # Load the dataset
    print("\nLoading and preprocessing the data...")
    dataset_path = os.path.join("/home/codespace/.cache/kagglehub/datasets/bittlingmayer/amazonreviews/versions/7", "train.ft.txt.bz2")  # Using the compressed file
    df = load_amazon_reviews(dataset_path, sample_size=20000)  # Using 20k reviews to avoid memory issues
    
    # Display basic information
    print("\nDataset Information:")
    print(f"Shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Check class distribution
    print("\nClass distribution:")
    print(df['sentiment'].value_counts())
    print(df['sentiment'].value_counts(normalize=True).map(lambda x: f"{x:.2%}"))
    
    # Preprocess the reviews
    print("\nPreprocessing reviews...")
    df['processed_review'] = df['review'].apply(preprocess_text)
    
    # Show sample of original vs processed text
    print("\nOriginal vs Processed Text:")
    for i in range(2):
        print(f"Original: {df['review'].iloc[i][:100]}...")
        print(f"Processed: {df['processed_review'].iloc[i][:100]}...")
        print()
    
    # Split data into training and testing sets
    X = df['processed_review']
    y = df['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # Initialize TF-IDF Vectorizer
    tfidf = TfidfVectorizer(
        max_features=2000,  # Reduced features for memory efficiency
        min_df=2,
        max_df=0.8,
        stop_words='english'
    )
    
    # Transform text to TF-IDF features
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    print(f"\nNumber of features: {X_train_tfidf.shape[1]}")
    
    # Train Logistic Regression model
    print("\nTraining the model...")
    model = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    
    model.fit(X_train_tfidf, y_train)
    
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
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix - Amazon Reviews')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('amazon_confusion_matrix.png')
    plt.close()
    
    # Analyze feature importance
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
    
    # Save the model and vectorizer
    print("\nSaving model and vectorizer...")
    import pickle
    with open('amazon_review_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('amazon_review_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    
    # Function to predict sentiment of new reviews
    def predict_sentiment(text):
        """Predict sentiment of new text"""
        processed = preprocess_text(text)
        text_tfidf = tfidf.transform([processed])
        prediction = model.predict(text_tfidf)[0]
        probability = model.predict_proba(text_tfidf)[0]
        return prediction, probability
    
    # Test with example reviews
    print("\nTesting the model with example reviews:")
    
    # Example 1: Positive review
    review = "This product exceeded my expectations! Great quality and fast shipping. Would definitely recommend!"
    prediction, probability = predict_sentiment(review)
    print("\nReview 1:", review)
    print(f"Prediction: {'Positive' if prediction == 1 else 'Negative'}")
    print(f"Confidence: {max(probability):.2%}")
    
    # Example 2: Negative review
    review = "Poor quality product. Broke after first use. Customer service was unhelpful. Don't waste your money."
    prediction, probability = predict_sentiment(review)
    print("\nReview 2:", review)
    print(f"Prediction: {'Positive' if prediction == 1 else 'Negative'}")
    print(f"Confidence: {max(probability):.2%}")
    
    # Example 3: Mixed review
    review = "Product is okay for the price. Some good features but also some drawbacks. Decent value overall."
    prediction, probability = predict_sentiment(review)
    print("\nReview 3:", review)
    print(f"Prediction: {'Positive' if prediction == 1 else 'Negative'}")
    print(f"Confidence: {max(probability):.2%}")
    
    print("\nAmazon review classification model completed!")

if __name__ == "__main__":
    main()