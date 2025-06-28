import pandas as pd
import spacy
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

if not os.path.exists('models'):
    os.makedirs('models')

# --- 1. Load Data ---
try:
    data = pd.read_csv("IMDB/IMDB Dataset.csv")
except FileNotFoundError:
    print("Error: 'IMDB/IMDB Dataset.csv' not found.")
    print("Please follow the setup instructions in README.md.")
    exit()

# --- 2. Prepare Data ---
print("Preparing data...")
nlp = spacy.load("en_core_web_sm")
stop_words = list(nlp.Defaults.stop_words)
X = data["review"]
y = data["sentiment"].apply(lambda s: 1 if s == "positive" else 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Define and Train Models ---
models = {
    "MultinomialNB": Pipeline([
        ('vectorizer', CountVectorizer(stop_words=stop_words)),
        ('classifier', MultinomialNB())
    ]),
    "KNeighborsClassifier": Pipeline([
        ('vectorizer', CountVectorizer(stop_words=stop_words)),
        ('classifier', KNeighborsClassifier(n_neighbors=10))
    ]),
    "RandomForest": Pipeline([
        ('vectorizer', CountVectorizer(stop_words=stop_words)),
        ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
    ]),
    "GradientBoosting": Pipeline([
        ('vectorizer', CountVectorizer(stop_words=stop_words)),
        ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ])
}

# Iterate through models to train, evaluate, and save them
for name, model in models.items():
    print("="*30)
    print(f"Training {name}...")
    if name == "GradientBoosting":
        print("(This may take a few minutes)")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predict = model.predict(X_test)
    print(f"\nClassification Report for {name}:\n")
    print(classification_report(y_test, predict))
    
    # Save the trained model
    model_path = f"models/{name}_pipeline.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}\n")

print("="*30)
print("Training complete. All models have been saved to the 'models/' directory.")