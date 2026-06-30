
import { removeAuthToken } from "../../../api/auth";
import { LOGOUT_EVENT } from "../../../components/main/AppHeader";

export const handleAuthError = (error, navigate) => {
  const msg = error?.message || "";

  if (/token|unauthorized|expired|invalid/i.test(msg)) {
    removeAuthToken();
    window.dispatchEvent(new Event(LOGOUT_EVENT));
    navigate("/login");
    return true;
  }

  return false;
};

export const cleanError = (msg) => {
  if (!msg) {
    return "Something went wrong. Please try again.";
  }

  const error = msg.toLowerCase();

  if (
    error.includes("does not support image") ||
    error.includes("cannot read")
  ) {
    return "Image files are not supported. Please upload a CSV, or Excel file.";
  }

  if (error.includes("failed to parse dataset")) {
    return "Could not read the file. Make sure it's a valid CSV, or Excel file.";
  }

  if (error.includes("dataset is required")) {
    return "No dataset found. Please upload a file first.";
  }

  if (
    error === "session_expired" ||
    error.includes("token") ||
    error.includes("unauthorized") ||
    error.includes("expired")
  ) {
    return "Your session has expired. Please log out and log in again.";
  }

  return msg;
};