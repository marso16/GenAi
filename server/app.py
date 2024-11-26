import pickle
import re
import numpy as np
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer

VECTOR_SIZE = 100

# Function to load GloVe embeddings
def load_glove_embeddings(file_path):
    print("Loading GloVe embeddings...")
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]  # The word
            vector = np.asarray(values[1:], dtype='float32')  # The embedding
            embeddings[word] = vector
    print(f"Loaded {len(embeddings)} words into embeddings.")
    return embeddings

# Function to compute average GloVe vector for a review
def get_average_glove_vector(review, embeddings, vector_size):
    words = review.split()
    vectors = []
    recognized_words = []  # Store recognized words

    for word in words:
        if word in embeddings:
            vectors.append(embeddings[word])
            recognized_words.append(word)

    if len(vectors) == 0:
        return np.zeros(vector_size), recognized_words
    
    return np.mean(vectors, axis=0), recognized_words

# Load GloVe embeddings
glove_embeddings = load_glove_embeddings("C:\\Users\\User\\Desktop\\cloned\\GenAi\\datasets\\glove.6B.100d.txt")

# Load pre-trained models and vectorizers
with open("./models/lr_bow.pkl", "rb") as f:
    lr_bow = pickle.load(f)
with open("./models/lr_tfidf.pkl", "rb") as f:
    lr_tfidf = pickle.load(f)
with open("./models/svm_bow.pkl", "rb") as f:
    svm_bow = pickle.load(f)
with open("./models/svm_tfidf.pkl", "rb") as f:
    svm_tfidf = pickle.load(f)
with open("./models/cv_vectorizer.pkl", "rb") as f:
    cv = pickle.load(f)
with open("./models/tfidf_vectorizer.pkl", "rb") as f:
    tv = pickle.load(f)

logistic_glove_embeddings = joblib.load('models/logistic_regression_model.pkl')
svm_glove_embeddings = joblib.load('models/svm_rbf_model.pkl')

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for input data
class ReviewRequest(BaseModel):
    review: str
    model: str = 'logistic'  # Options: 'logistic', 'svm'
    vectorizer: str = 'tfidf'  # Options: 'tfidf', 'bow', 'glove_embeddings'

# Preprocessing functions
def denoise_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    return re.sub(r'\[[^]]*\]', '', text)

def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-Z0-9\s]'
    return re.sub(pattern, '', text)

def simple_lemmatizer(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stopwords(text):
    tokenizer = ToktokTokenizer()
    stopword_list = stopwords.words('english')
    tokens = tokenizer.tokenize(text)
    return ' '.join([token for token in tokens if token.lower() not in stopword_list])

# Prediction function
def predict_sentiment(review, model='logistic', vectorizer='tfidf'):
    # Preprocess review
    review = denoise_text(review)
    review = remove_special_characters(review)
    review = remove_stopwords(review)
    review = simple_lemmatizer(review)

    # Transform review using the chosen vectorizer
    if vectorizer == 'tfidf':
        review_vector = tv.transform([review])
    elif vectorizer == 'bow':
        review_vector = cv.transform([review])
    elif vectorizer == 'glove_embeddings':
        review_vector, recognized_words = get_average_glove_vector(review, glove_embeddings, VECTOR_SIZE)
        review_vector = review_vector.reshape(1, -1)
    else:
        raise ValueError("Vectorizer must be 'tfidf', 'bow', or 'glove_embeddings'")

    # Predict using the chosen model
    if model == 'logistic':
        if vectorizer == 'tfidf':
            prediction = lr_tfidf.predict(review_vector)
        elif vectorizer == 'bow':
            prediction = lr_bow.predict(review_vector)
        elif vectorizer == 'glove_embeddings':
            prediction = logistic_glove_embeddings.predict(review_vector)
    elif model == 'svm':
        if vectorizer == 'tfidf':
            prediction = svm_tfidf.predict(review_vector)
        elif vectorizer == 'bow':
            prediction = svm_bow.predict(review_vector)
        elif vectorizer == 'glove_embeddings':
            prediction = svm_glove_embeddings.predict(review_vector)
    else:
        raise ValueError("Model must be 'logistic' or 'svm'")

    sentiment = 'Positive' if prediction[0].lower() == 'positive' else 'Negative'

    return sentiment, recognized_words

# Endpoint for predicting sentiment
@app.post("/predict/")
def get_sentiment(request: ReviewRequest):
    sentiment, recognized_words = predict_sentiment(request.review, model=request.model, vectorizer=request.vectorizer)
    return {
        "review": request.review,
        "sentiment": sentiment,
        "recognized_words": recognized_words
    }
