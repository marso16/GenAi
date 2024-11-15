import React from "react";

const LoadingSpinner = () => {
  return (
    <div className="flex justify-center items-center my-4">
      <div className="spinner-border animate-spin inline-block w-8 h-8 border-4 border-solid border-orange-600 rounded-full" />
    </div>
  );
};

export default LoadingSpinner;
