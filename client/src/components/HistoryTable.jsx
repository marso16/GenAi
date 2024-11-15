import React from "react";

const HistoryTable = ({ reviewHistory, handleSelectReview }) => {
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full table-auto">
        <thead>
          <tr className="bg-gray-100">
            <th className="px-4 py-2 text-left">Action</th>
            <th className="px-4 py-2 text-left">Review</th>
            <th className="px-4 py-2 text-left">Sentiment</th>
            <th className="px-4 py-2 text-left">Model</th>
            <th className="px-4 py-2 text-left">Vectorizer</th>
          </tr>
        </thead>
        <tbody>
          {reviewHistory.length === 0 ? (
            <tr>
              <td colSpan="5" className="px-4 py-2 text-center text-gray-500">
                No history available.
              </td>
            </tr>
          ) : (
            reviewHistory.map((item, index) => (
              <tr key={index} className="border-t">
                <td className="px-4 py-2">
                  <button
                    className="text-blue-600"
                    onClick={() => handleSelectReview(item)}
                  >
                    Select
                  </button>
                </td>
                <td className="px-4 py-2">{item.text}</td>
                <td className="px-4 py-2">{item.sentiment}</td>
                <td className="px-4 py-2">{item.model}</td>
                <td className="px-4 py-2">{item.vectorizer}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
};

export default HistoryTable;
