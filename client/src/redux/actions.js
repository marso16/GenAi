export const SET_REVIEW = "SET_REVIEW";
export const SET_MODEL = "SET_MODEL";
export const SET_VECTORIZER = "SET_VECTORIZER";
export const SET_RESULT = "SET_RESULT";

export const setReview = (review) => ({
  type: SET_REVIEW,
  payload: review,
});

export const setModel = (model) => ({
  type: SET_MODEL,
  payload: model,
});

export const setVectorizer = (vectorizer) => ({
  type: SET_VECTORIZER,
  payload: vectorizer,
});

export const setResult = (result) => ({
  type: SET_RESULT,
  payload: result,
});
