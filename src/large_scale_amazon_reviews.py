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
import pickle

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
from sklearn.metrics import roc_curve, auc
from sklearn.base import BaseEstimator, TransformerMixin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Custom transformer for text preprocessing"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return [self.preprocess_text(text) for text in X]
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize, remove stopwords, and lemmatize
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        
        # Join words back into a string
        return ' '.join(words)

def count_lines(file_path):
    """Count the total number of lines in a bz2 compressed file"""
    count = 0
    with bz2.open(file_path, 'rt', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count

def load_amazon_reviews_in_chunks(file_path, chunk_size=100000, total_samples=None):
    """
    Load and preprocess Amazon reviews data in chunks to manage memory
    
    Parameters:
    file_path (str): Path to the dataset
    chunk_size (int): Number of reviews to process at once
    total_samples (int): Total number of samples to process (None for all)
    
    Yields:
    pd.DataFrame: Chunks of processed reviews
    """
    reviews = []
    labels = []
    count = 0
    
    logging.info(f"Reading reviews in chunks of {chunk_size}")
    with bz2.open(file_path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, total=total_samples):
            if not line.strip():
                continue
                
            # Extract label and review
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                label, review = parts
                labels.append(label)
                reviews.append(review)
                count += 1
                
                # Yield chunk when it reaches chunk_size
                if len(reviews) >= chunk_size:
                    df_chunk = pd.DataFrame({
                        'sentiment': labels,
                        'review': reviews
                    })
                    df_chunk['sentiment'] = (df_chunk['sentiment'] == '__label__2').astype(int)
                    df_chunk = df_chunk.dropna()
                    
                    yield df_chunk
                    
                    # Clear lists and garbage collect
                    reviews = []
                    labels = []
                    gc.collect()
                
            if total_samples and count >= total_samples:
                break
    
    # Yield remaining reviews
    if reviews:
        df_chunk = pd.DataFrame({
            'sentiment': labels,
            'review': reviews
        })
        df_chunk['sentiment'] = (df_chunk['sentiment'] == '__label__2').astype(int)
        df_chunk = df_chunk.dropna()
        yield df_chunk

def plot_roc_curve(y_test, y_pred_proba, output_path='roc_curve.png'):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    plt.close()

def main():
    # Download NLTK resources
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    
    # File path to the dataset
    dataset_path = "/home/codespace/.cache/kagglehub/datasets/bittlingmayer/amazonreviews/versions/7/train.ft.txt.bz2"
    
    # Count total lines
    total_lines = count_lines(dataset_path)
    logging.info(f"Total number of reviews: {total_lines}")
    
    # Initialize models
    preprocessor = TextPreprocessor()
    vectorizer = TfidfVectorizer(
        max_features=5000,  # Reduced for memory efficiency
        min_df=10,
        max_df=0.8,
        ngram_range=(1, 2)  # Include bigrams
    )
    classifier = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    
    # Process data in chunks (reduced for initial testing)
    chunk_size = 10000
    total_samples = 100000  # Process 100K reviews for this run
    
    # Lists to store all preprocessed text and labels
    all_processed_text = []
    all_labels = []
    
    logging.info("Processing reviews...")
    for chunk in load_amazon_reviews_in_chunks(dataset_path, chunk_size, total_samples):
        # Preprocess text
        processed_chunk = preprocessor.transform(chunk['review'])
        all_processed_text.extend(processed_chunk)
        all_labels.extend(chunk['sentiment'])
        
        # Free memory
        gc.collect()
    
    # Convert to numpy arrays
    X = np.array(all_processed_text)
    y = np.array(all_labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Free memory
    del X, y
    gc.collect()
    
    # Transform text data
    logging.info("Vectorizing text...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train model
    logging.info("Training model...")
    classifier.fit(X_train_tfidf, y_train)
    
    # Make predictions
    y_pred = classifier.predict(X_test_tfidf)
    y_pred_proba = classifier.predict_proba(X_test_tfidf)[:, 1]
    
    # Calculate and display metrics
    accuracy = np.mean(y_pred == y_test)
    logging.info(f"\nAccuracy: {accuracy:.4f}")
    
    logging.info("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logging.info("\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix - Amazon Reviews')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('large_scale_confusion_matrix.png')
    plt.close()
    
    # Plot ROC curve
    plot_roc_curve(y_test, y_pred_proba)
    
    # Analyze feature importance
    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]
    
    # Create DataFrame of feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': coefficients
    })
    
    # Sort by absolute importance
    feature_importance['Abs_Importance'] = abs(feature_importance['Importance'])
    feature_importance = feature_importance.sort_values('Abs_Importance', ascending=False)
    
    # Display top features
    logging.info("\nTop 20 Most Important Features:")
    print(feature_importance.head(20))
    
    # Save feature importance
    feature_importance.head(100).to_csv('feature_importance.csv', index=False)
    
    # Save models
    logging.info("\nSaving models...")
    with open('large_scale_amazon_model.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    
    with open('large_scale_amazon_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Test with example reviews
    def predict_sentiment(text):
        """Predict sentiment of new text"""
        processed = preprocessor.transform([text])[0]
        text_tfidf = vectorizer.transform([processed])
        prediction = classifier.predict(text_tfidf)[0]
        probability = classifier.predict_proba(text_tfidf)[0]
        return prediction, probability
    
    logging.info("\nTesting model with example reviews:")
    
    # Example 1: Positive review
    review = "This product exceeded my expectations! Great quality and fast shipping. Would definitely recommend!"
    prediction, probability = predict_sentiment(review)
    logging.info("\nReview 1: %s", review)
    logging.info("Prediction: %s", 'Positive' if prediction == 1 else 'Negative')
    logging.info("Confidence: %.2f%%", max(probability) * 100)
    
    # Example 2: Negative review
    review = "Poor quality product. Broke after first use. Customer service was unhelpful. Don't waste your money."
    prediction, probability = predict_sentiment(review)
    logging.info("\nReview 2: %s", review)
    logging.info("Prediction: %s", 'Positive' if prediction == 1 else 'Negative')
    logging.info("Confidence: %.2f%%", max(probability) * 100)
    
    # Example 3: Mixed review
    review = "Product is okay for the price. Some good features but also some drawbacks. Decent value overall."
    prediction, probability = predict_sentiment(review)
    logging.info("\nReview 3: %s", review)
    logging.info("Prediction: %s", 'Positive' if prediction == 1 else 'Negative')
    logging.info("Confidence: %.2f%%", max(probability) * 100)
    
    logging.info("\nLarge-scale Amazon review classification completed!")

if __name__ == "__main__":
    main()