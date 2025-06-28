# IMDB Movie Review Sentiment Analysis

This project performs sentiment analysis on the IMDB 50k Movie Review dataset. It trains and evaluates four different machine learning models to classify movie reviews as either "positive" or "negative".

The trained models are:
-   Multinomial Naive Bayes
-   K-Nearest Neighbors (KNN)
-   Random Forest
-   Gradient Boosting

## Project Structure

Your project directory should be set up as follows for the scripts to work correctly.

```
sentiment_project/
|
├── IMDB/
|   └── IMDB Dataset.csv      <-- The dataset file
|
├── models/                   <-- Created by train.py to store models
|
├── train.py                  <-- Script to train and save all models
├── predict.py                <-- Script to run predictions using a saved model
├── requirements.txt          <-- List of Python dependencies
└── README.md                 <-- This file
```

## Setup and Installation

### Prerequisites
-   Python 3.8+

### Step-by-Step Guide

1.  **Clone the Repository (or create your project folder)**
    ```bash
    git clone <your-repo-url>
    cd sentiment_project
    ```

2.  **Download the Dataset**
    -   Download the dataset from Kaggle: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
    -   Unzip the file and place `IMDB Dataset.csv` inside a folder named `IMDB` within your project directory.

3.  **Create and Activate a Virtual Environment**
    It is highly recommended to use a virtual environment to keep dependencies isolated.

    *   **On Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    *   **On macOS / Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

4.  **Install Dependencies**
    Install all the required packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

5.  **Download the SpaCy Language Model**
    The script uses a small English model from `spacy` for stop word removal.
    ```bash
    python -m spacy download en_core_web_sm
    ```

## How to Use

### 1. How to Train the Models
To train all four models, run the `train.py` script. This will process the data, train each model, print its classification report, and save the final trained pipeline to the `models/` directory.

```bash
python train.py
```
This process might take several minutes, especially for the Gradient Boosting model.

### 2. How to Run Predictions
After training, you can use the `predict.py` script to get sentiment predictions for new sentences.

-   You can change the model used for prediction by editing the `MODEL_NAME` variable inside `predict.py`.
-   You can also change the `sample_reviews` list to test your own sentences.

Run the script from your terminal:
```bash
python predict.py
```

## Model Performance

The following table summarizes the accuracy of each model on the test set after running the `train.py` script. Accuracy represents the percentage of reviews that were correctly classified.

| Model                  | Accuracy | F1-Score (Macro Avg) |
|------------------------|:--------:|:--------------------:|
| **Multinomial Naive Bayes** |   86%    |         0.86         |
| **Random Forest**          |   85%    |         0.85         |
| **Gradient Boosting**      |   81%    |         0.81         |
| **K-Nearest Neighbors**    |   61%    |         0.59         |

### Analysis
-   **Top Performers**: **Multinomial Naive Bayes** and **Random Forest** show the strongest performance, both achieving around 85-86% accuracy. This is excellent for a baseline model using a simple Bag-of-Words approach.
-   **Poor Performer**: **K-Nearest Neighbors (KNN)** struggles significantly. This is expected, as KNN is sensitive to the "curse of dimensionality." Text data converted with `CountVectorizer` creates thousands of dimensions (one for each word), and distance-based algorithms like KNN do not perform well in such a high-dimensional, sparse space.

### A Note on the `UserWarning`
During training, you may see a `UserWarning: Your stop_words may be inconsistent with your preprocessing`. This occurs because `CountVectorizer`'s default tokenizer splits contractions like "don't" into "don" and "t", while the stop word list from `spacy` contains "n't". This mismatch is minor and has a negligible impact on the final performance of these models.