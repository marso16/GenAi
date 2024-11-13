import { configureStore } from "@reduxjs/toolkit";
import { rootReducer } from "./reducer";

// Create store
export const store = configureStore({
  reducer: rootReducer,
});
