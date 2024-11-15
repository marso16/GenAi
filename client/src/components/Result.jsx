import React from "react";

const Result = ({ formSubmitted, review, result }) => {
  const resultColor =
    result === "Positive"
      ? "bg-green-500 text-white"
      : result === "Negative"
      ? "bg-red-500 text-white"
      : "bg-gray-100 text-gray-700";

  return (
    <>
      {formSubmitted && !review && (
        <div className="mt-6 p-4 text-center rounded border bg-yellow-100 text-yellow-700">
          <span className="font-semibold">Note:</span> No review entered.
        </div>
      )}
      {result && review && (
        <div className={`mt-6 p-4 text-center rounded border ${resultColor}`}>
          <span className="font-semibold">Sentiment:</span> {result}
        </div>
      )}
    </>
  );
};

export default Result;
