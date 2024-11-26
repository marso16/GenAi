export const API_URL = "http://127.0.0.1:8000/predict";

const MODEL_OPTIONS = [
  { value: "logistic", label: "Logistic Regression" },
  { value: "svm", label: "SVM" },
];

const VECTORIZER_OPTIONS = [
  { value: "tfidf", label: "TF-IDF" },
  { value: "bow", label: "BoW" },
];

export { MODEL_OPTIONS, VECTORIZER_OPTIONS };
