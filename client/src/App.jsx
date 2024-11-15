import React from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  MODEL_OPTIONS,
  VECTORIZER_OPTIONS,
  API_URL,
} from "./constants/constants";
import {
  setReview,
  setModel,
  setVectorizer,
  setResult,
  addToHistory,
} from "./redux/actions";
import HistoryTable from "./components/HistoryTable";

const App = () => {
  const dispatch = useDispatch();

  const review = useSelector((state) => state.review);
  const selectedModel = useSelector((state) => state.selectedModel);
  const selectedVectorizer = useSelector((state) => state.selectedVectorizer);
  const result = useSelector((state) => state.result);
  const reviewHistory = useSelector((state) => state.reviewHistory);

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

      dispatch(
        addToHistory({
          text: review,
          sentiment: data.sentiment,
          model: selectedModel,
          vectorizer: selectedVectorizer,
        })
      );
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

  const handleSelectReview = (item) => {
    dispatch(setReview(item.text));
    dispatch(setModel(item.model));
    dispatch(setVectorizer(item.vectorizer));
  };

  return (
    <div className="min-h-screen bg-gray-100 flex items-start justify-center p-4">
      <div className="flex space-x-8 w-full max-w-6xl">
        {/* Left: Input Form */}
        <div className="bg-white shadow-md rounded px-8 py-6 w-2/3">
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
                {VECTORIZER_OPTIONS.map((option) => {
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>;
                })}
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
            <div
              className={`mt-6 p-4 text-center rounded border ${resultColor}`}
            >
              <span className="font-semibold">Sentiment:</span> {result}
            </div>
          )}
        </div>

        {/* Right: History Table */}
        <div className="bg-white shadow-md rounded px-8 py-6 w-1/3">
          <h2 className="text-xl font-semibold text-center mb-4">
            Review History
          </h2>
          <HistoryTable
            reviewHistory={reviewHistory}
            handleSelectReview={handleSelectReview}
          />
        </div>
      </div>
    </div>
  );
};

export default App;
