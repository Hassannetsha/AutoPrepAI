const API_BASE_URL =
  process.env.REACT_APP_API_BASE_URL || "http://localhost:8022";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
    ...options,
  });

  let data = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }

  if (!response.ok) {
    const message =
      data?.detail ||
      data?.message ||
      "Something went wrong. Please try again.";
    throw new Error(
      Array.isArray(message)
        ? message.map((item) => item.msg).join(" ")
        : message,
    );
  }

  return data;
}

export function loginUser({ email, password }) {
  return request("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export function signupUser({
  email,
  password,
  confirmPassword,
  firstName,
  lastName,
  phoneNumber,
  
}) {
  return request("/auth/signup", {
    method: "POST",
    body: JSON.stringify({
      email,
      password,
      confirm_password: confirmPassword,
      first_name: firstName,
      last_name: lastName,
      phone_number: phoneNumber || null,
    }),
  });
}

export function verifyEmailToken(token) {
  return request(`/auth/verify-email?token=${encodeURIComponent(token)}`);
}

export function resendVerificationEmail(email) {
  return request("/auth/resend-verification", {
    method: "POST",
    body: JSON.stringify({ email }),
  });
}

export function forgotPassword({ email }) {
  return request("/auth/forgot-password", {
    method: "POST",
    body: JSON.stringify({ email }),
  });
}

export function resetPassword({ token, newPassword }) {
  return request("/auth/reset-password", {
    method: "POST",
    body: JSON.stringify({ token, new_password: newPassword }),
  });
}

export function storeAuthToken(accessToken) {
  localStorage.setItem("autoprepai_access_token", accessToken);
}

export function getAuthToken() {
  return localStorage.getItem("autoprepai_access_token");
}

export function removeAuthToken() { // for logout
  localStorage.removeItem("autoprepai_access_token");
}
