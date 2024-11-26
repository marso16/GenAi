import pickle
import re
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

VECTOR_SIZE = 100

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

def get_average_glove_vector(review, embeddings, vector_size):
    """
    Compute the average GloVe vector for a given review.
    
    :param review: A string representing the review
    :param embeddings: Dictionary of GloVe word embeddings
    :param vector_size: Dimension of the embeddings (e.g., 100 for glove.6B.100d.txt)
    :return: A numpy array representing the average vector for the review
    """
    words = review.split()
    vectors = []
    
    for word in words:
        if word in embeddings:
            vectors.append(embeddings[word])
    
    if len(vectors) == 0:
        # Return a zero vector if no words are found
        return np.zeros(vector_size)
    
    # Compute the average vector
    return np.mean(vectors, axis=0)

glove_embeddings = load_glove_embeddings("embeddings/glove.6B.100d.txt")
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

logistic_glove_embeddings = joblib.load('models/logistic_regression_model.pkl')
svm_glove_embeddings = joblib.load('models/svm_rbf_model.pkl')

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
    review = remove_stopwords(review)
    review = simple_stemmer(review)
    

    # Transform review using the chosen vectorizer
    if vectorizer == 'tfidf':
        review_vector = tv.transform([review])
    elif vectorizer == 'bow':
        review_vector = cv.transform([review])
    elif vectorizer == "glove_embeddings":
        review_vector = get_average_glove_vector(review, glove_embeddings, VECTOR_SIZE)
    else:
        raise ValueError("Vectorizer must be either 'tfidf' or 'bow'")

    # Predict using the chosen model
    if model == 'logistic':
        prediction = lr_tfidf.predict(review_vector) if vectorizer == 'tfidf' else lr_bow.predict(review_vector)
    elif model == 'svm':
        prediction = svm_tfidf.predict(review_vector) if vectorizer == 'tfidf' else svm_bow.predict(review_vector)
    elif model == 'naive_bayes':
        prediction = mnb_tfidf.predict(review_vector) if vectorizer == 'tfidf' else mnb_bow.predict(review_vector)
    elif model == 'svm_glove':
        prediction == svm_glove_embeddings.predict(review_vector)
    elif model == 'logistic_glove':
        prediction == logistic_glove_embeddings.predict(review_vector)
    else:
        raise ValueError("Model must be either 'logistic', 'svm', or 'naive_bayes'")

    return 'Positive' if prediction == 1 else 'Negative'

# Endpoint for predicting sentiment
@app.post("/predict/")
def get_sentiment(request: ReviewRequest):
    sentiment = predict_sentiment(request.review, model=request.model, vectorizer=request.vectorizer)
    return {"review": request.review, "sentiment": sentiment}
