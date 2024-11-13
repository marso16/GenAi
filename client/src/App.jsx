import React from "react";
import { useDispatch, useSelector } from "react-redux";
import axios from "axios";
import { setReview, setModel, setVectorizer, setResult } from "./redux/actions";

const App = () => {
  const dispatch = useDispatch();
  const { review, selectedModel, selectedVectorizer, result } = useSelector(
    (state) => state
  );

  const modelOptions = [
    { value: "logistic", label: "Logistic Regression" },
    { value: "svm", label: "SVM" },
    { value: "naive_bayes", label: "Naive Bayes" },
  ];

  const vectorizerOptions = [
    { value: "tfidf", label: "TF-IDF" },
    { value: "bow", label: "Bag of Words" },
  ];

  const handleReviewChange = (e) => {
    dispatch(setReview(e.target.value));
  };

  const handleModelChange = (e) => {
    dispatch(setModel(e.target.value));
  };

  const handleVectorizerChange = (e) => {
    dispatch(setVectorizer(e.target.value));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/predict",
        {
          review,
          model: selectedModel,
          vectorizer: selectedVectorizer,
        },
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (response.status === 200) {
        dispatch(setResult(response.data.sentiment));
      } else {
        dispatch(setResult("Error: Unable to analyze sentiment"));
      }
    } catch (error) {
      dispatch(setResult("Error: Network issue or server not available"));
    }
  };

  const resultColor =
    result === "Positive"
      ? "bg-green-500 text-white"
      : result === "Negative"
      ? "bg-red-500 text-white"
      : "bg-gray-100 text-gray-700";

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
      <div className="bg-white shadow-md rounded px-8 py-6 max-w-lg w-full">
        <div className="flex items-center justify-center mb-4">
          <h1 className="text-2xl font-bold text-center">Sentiment Analysis</h1>
        </div>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-gray-700">Review</label>
            <textarea
              value={review}
              onChange={handleReviewChange}
              className="mt-1 w-full p-2 border border-gray-300 rounded focus:outline-none focus:ring focus:ring-blue-300"
              rows="4"
              placeholder="Enter your review here"
            ></textarea>
          </div>

          <div>
            <label className="block text-gray-700">Model</label>
            <select
              value={selectedModel}
              onChange={handleModelChange}
              className="mt-1 w-full p-2 border border-gray-300 rounded focus:outline-none focus:ring focus:ring-blue-300"
            >
              {modelOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-gray-700">Vectorizer</label>
            <select
              value={selectedVectorizer}
              onChange={handleVectorizerChange}
              className="mt-1 w-full p-2 border border-gray-300 rounded focus:outline-none focus:ring focus:ring-blue-300"
            >
              {vectorizerOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <button
            type="submit"
            className="w-full bg-orange-600 text-white p-2 rounded hover:bg-blue-600 focus:outline-none focus:ring focus:ring-blue-300"
          >
            Analyze Sentiment
          </button>
        </form>

        {result && (
          <div className={`mt-6 p-4 text-center rounded border ${resultColor}`}>
            <span className="font-semibold">Sentiment:</span> {result}
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
