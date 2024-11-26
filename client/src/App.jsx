import React, { useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  MODEL_OPTIONS,
  VECTORIZER_OPTIONS,
  API_URL,
} from "./constants/constants";
import { setReview, setModel, setVectorizer, setResult } from "./redux/actions";

const App = () => {
  const dispatch = useDispatch();
  const [error, setError] = useState("");

  const review = useSelector((state) => state.review);
  const selectedModel = useSelector((state) => state.selectedModel);
  const selectedVectorizer = useSelector((state) => state.selectedVectorizer);
  const result = useSelector((state) => state.result);

  const handleReviewChange = (e) => {
    const value = e.target.value;
    dispatch(setReview(value));
    setError(""); // Clear error as user types
  };

  const handleModelChange = (e) => {
    dispatch(setModel(e.target.value));
  };

  const handleVectorizerChange = (e) => {
    dispatch(setVectorizer(e.target.value));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Validate input
    if (!review.trim()) {
      setError("Review cannot be empty.");
      return;
    }

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          review,
          model: selectedModel,
          vectorizer: selectedVectorizer,
        }),
      });

      const data = await response.json();
      dispatch(setResult(data.sentiment));
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
    <div className="min-h-screen bg-gray-100 flex items-center justify-center">
      <div className="bg-white shadow-md rounded px-8 py-6 w-full max-w-lg">
        <h1 className="text-2xl font-bold text-center mb-6">
          Sentiment Analysis
        </h1>

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
              {MODEL_OPTIONS.map((option) => (
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
              {VECTORIZER_OPTIONS.map((option) => (
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

        {error && (
          <div className="mt-4 p-4 text-center bg-red-100 text-red-700 rounded border border-red-300">
            {error}
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
