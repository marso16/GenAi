import { SET_REVIEW, SET_MODEL, SET_VECTORIZER, SET_RESULT } from "./actions";

// Initial state
const initialState = {
  review: "",
  selectedModel: "logistic",
  selectedVectorizer: "tfidf",
  result: "",
};

// Reducer function
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
    default:
      return state;
  }
};
