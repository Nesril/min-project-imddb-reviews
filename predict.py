import joblib
import os

# Choose which model to use for prediction by changing the filename
MODEL_NAME = "RandomForest_pipeline.joblib" 
MODEL_PATH = os.path.join("models", MODEL_NAME)

# --- Prediction Function ---
def predict_sentiment(text_list, model_path):
    if not os.path.exists(model_path):
        return f"Error: Model file not found at {model_path}. Please run train.py first."
        
    model = joblib.load(model_path)
    
    predictions = model.predict(text_list)
    
    labels = ["Negative" if p == 0 else "Positive" for p in predictions]
    
    return labels

if __name__ == "__main__":
    # Example sentences to predict
    sample_reviews = [
        "This movie was absolutely fantastic, a must-see!",
        "I was really disappointed. The plot was weak and the acting was terrible.",
        "It was an okay film, not great but not bad either.",
        "The special effects were breathtaking, but the story was predictable."
    ]
    
    results = predict_sentiment(sample_reviews, MODEL_PATH)
    
    if isinstance(results, str): 
        print(results)
    else:
        print(f"--- Predictions using {MODEL_NAME} ---")
        for review, sentiment in zip(sample_reviews, results):
            print(f"Review: '{review}'\nSentiment: {sentiment}\n")