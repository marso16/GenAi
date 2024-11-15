import {
  SET_REVIEW,
  SET_MODEL,
  SET_VECTORIZER,
  SET_RESULT,
  ADD_TO_HISTORY,
} from "./actions";

const initialState = {
  review: "",
  selectedModel: "logistic",
  selectedVectorizer: "tfidf",
  result: "",
  reviewHistory: [],
};

export const rootReducer = (state = initialState, action) => {
  switch (action.type) {
    case SET_REVIEW:
      return { ...state, review: action.payload };
    case SET_MODEL:
      return { ...state, selectedModel: action.payload };
    case SET_VECTORIZER:
      return { ...state, selectedVectorizer: action.payload };
    case SET_RESULT:
      return { ...state, result: action.payload };
    case ADD_TO_HISTORY:
      return {
        ...state,
        reviewHistory: [
          ...state.reviewHistory,
          {
            text: action.payload.text,
            sentiment: action.payload.sentiment,
            model: action.payload.model,
            vectorizer: action.payload.vectorizer,
          },
        ],
      };
    default:
      return state;
  }
};
