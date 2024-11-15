import React from "react";
import { useDispatch, useSelector } from "react-redux";
import { setReview, setModel, setVectorizer } from "../redux/actions";
import { MODEL_OPTIONS, VECTORIZER_OPTIONS } from "../constants/constants";

const Form = ({ onSubmit }) => {
  const dispatch = useDispatch();

  const review = useSelector((state) => state.review);
  const selectedModel = useSelector((state) => state.selectedModel);
  const selectedVectorizer = useSelector((state) => state.selectedVectorizer);

  const handleReviewChange = (e) => {
    dispatch(setReview(e.target.value));
  };

  const handleModelChange = (e) => {
    dispatch(setModel(e.target.value));
  };

  const handleVectorizerChange = (e) => {
    dispatch(setVectorizer(e.target.value));
  };

  return (
    <form onSubmit={onSubmit} className="space-y-4">
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
  );
};

export default Form;
