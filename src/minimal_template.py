# STEP 1: Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# STEP 2: Load and prepare data
df = pd.read_csv('data/spam.csv', encoding='latin1')
print("\nColumn Names:")
print(df.columns.tolist())

X = df['v2']  # text message column
y = df['v1']  # target column (spam/ham)

# STEP 3: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 4: Feature engineering with TF-IDF
tfidf = TfidfVectorizer(max_features=5000, 
                        stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# STEP 5: Train logistic regression model
model = LogisticRegression(class_weight='balanced')
model.fit(X_train_tfidf, y_train)

# STEP 6: Evaluate model
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))
