import pickle
import re
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load the pre-trained models and vectorizers
with open("./models/lr_bow.pkl", "rb") as f:
    lr_bow = pickle.load(f)
with open("./models/lr_tfidf.pkl", "rb") as f:
    lr_tfidf = pickle.load(f)
with open("./models/svm_bow.pkl", "rb") as f:
    svm_bow = pickle.load(f)
with open("./models/svm_tfidf.pkl", "rb") as f:
    svm_tfidf = pickle.load(f)
with open("./models/mnb_bow.pkl", "rb") as f:
    mnb_bow = pickle.load(f)
with open("./models/mnb_tfidf.pkl", "rb") as f:
    mnb_tfidf = pickle.load(f)
with open("./models/cv_vectorizer.pkl", "rb") as f:
    cv = pickle.load(f)
with open("./models/tfidf_vectorizer.pkl", "rb") as f:
    tv = pickle.load(f)

# Define FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a Pydantic model for input data
class ReviewRequest(BaseModel):
    review: str
    model: str = 'logistic'  # Options: 'logistic', 'svm', 'naive_bayes'
    vectorizer: str = 'tfidf'  # Options: 'tfidf', 'bow'

# Function for preprocessing and prediction
def predict_sentiment(review, model='logistic', vectorizer='tfidf'):
    # Preprocessing functions (reuse from your original code)
    def denoise_text(text):
        from bs4 import BeautifulSoup
        text = BeautifulSoup(text, "html.parser").get_text()
        return re.sub(r'\[[^]]*\]', '', text)

    def remove_special_characters(text, remove_digits=True):
        pattern = r'[^a-zA-Z0-9\s]'
        return re.sub(pattern, '', text)

    def simple_stemmer(text):
        from nltk.stem import PorterStemmer
        ps = PorterStemmer()
        return ' '.join([ps.stem(word) for word in text.split()])

    def remove_stopwords(text):
        from nltk.corpus import stopwords
        from nltk.tokenize.toktok import ToktokTokenizer
        tokenizer = ToktokTokenizer()
        stopword_list = stopwords.words('english')
        tokens = tokenizer.tokenize(text)
        return ' '.join([token for token in tokens if token.lower() not in stopword_list])

    # Apply preprocessing
    review = denoise_text(review)
    review = remove_special_characters(review)
    review = simple_stemmer(review)
    review = remove_stopwords(review)

    # Transform review using the chosen vectorizer
    if vectorizer == 'tfidf':
        review_vector = tv.transform([review])
    elif vectorizer == 'bow':
        review_vector = cv.transform([review])
    else:
        raise ValueError("Vectorizer must be either 'tfidf' or 'bow'")

    # Predict using the chosen model
    if model == 'logistic':
        prediction = lr_tfidf.predict(review_vector) if vectorizer == 'tfidf' else lr_bow.predict(review_vector)
    elif model == 'svm':
        prediction = svm_tfidf.predict(review_vector) if vectorizer == 'tfidf' else svm_bow.predict(review_vector)
    elif model == 'naive_bayes':
        prediction = mnb_tfidf.predict(review_vector) if vectorizer == 'tfidf' else mnb_bow.predict(review_vector)
    else:
        raise ValueError("Model must be either 'logistic', 'svm', or 'naive_bayes'")

    return 'Positive' if prediction == 1 else 'Negative'

# Endpoint for predicting sentiment
@app.post("/predict/")
def get_sentiment(request: ReviewRequest):
    sentiment = predict_sentiment(request.review, model=request.model, vectorizer=request.vectorizer)
    return {"review": request.review, "sentiment": sentiment}
