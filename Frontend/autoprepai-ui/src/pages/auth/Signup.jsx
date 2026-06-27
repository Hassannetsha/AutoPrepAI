import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Database } from "lucide-react";
import { signupUser } from "../../api/auth";
import CountryPhoneInput from "../../components/auth/CountryPhoneInput";
import "../../styles/Auth.css";

function Signup() {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    firstName: "",
    lastName: "",
    email: "",
    phoneNumber: "",
    password: "",
    confirmPassword: "",
  });
  const [error, setError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const PHONE_REGEX = /^\+20(1[0-9]{9}|[2-9][0-9]{7,8})$/;

  const handleChange = (event) => {
    const { name, value } = event.target;
    setForm((currentForm) => ({ ...currentForm, [name]: value }));
    setError("");
    setSuccessMessage("");
  };

  const handlePhoneChange = (value) => {
    setForm((currentForm) => ({ ...currentForm, phoneNumber: value }));
    setError("");
    setSuccessMessage("");
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!form.firstName.trim() || !form.lastName.trim() || !form.email.trim() || !form.password) {
      setError("Please fill in your name, email, and password.");
      return;
    }

    if (form.password !== form.confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    if (form.phoneNumber && !PHONE_REGEX.test(form.phoneNumber)) {
      setError("Phone number must be a valid Egyptian number starting with +20 (e.g., +2010XXXXXXXX).");
      return;
    }

    setIsSubmitting(true);
    setError("");
    setSuccessMessage("");

    try {
      const signupEmail = form.email.trim();
      const data = await signupUser({
        firstName: form.firstName.trim(),
        lastName: form.lastName.trim(),
        email: signupEmail,
        phoneNumber: form.phoneNumber.trim(),
        password: form.password,
        confirmPassword: form.confirmPassword,
      });

      setSuccessMessage(data.message || "Account created. Please check your email to verify your account.");
      setForm({
        firstName: "",
        lastName: "",
        email: "",
        phoneNumber: "",
        password: "",
        confirmPassword: "",
      });
      navigate(`/verify-email?email=${encodeURIComponent(signupEmail)}`);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="auth-wrapper">
      <form className="auth-container" onSubmit={handleSubmit}>
        <div className="icon">
          <Database size={32} />
        </div>

        <h2>Sign Up</h2>
        <p className="subtitle">Create your account</p>

        <div className="input-group">
          <label>First name</label>
          <input
            name="firstName"
            value={form.firstName}
            onChange={handleChange}
            placeholder="Enter your first name"
            autoComplete="given-name"
          />
        </div>

        <div className="input-group">
          <label>Last name</label>
          <input
            name="lastName"
            value={form.lastName}
            onChange={handleChange}
            placeholder="Enter your last name"
            autoComplete="family-name"
          />
        </div>

        <div className="input-group">
          <label>Email</label>
          <input
            name="email"
            type="email"
            value={form.email}
            onChange={handleChange}
            placeholder="Enter your email"
            autoComplete="email"
          />
        </div>

        <div className="input-group">
          <label>Phone number</label>
          <CountryPhoneInput
            value={form.phoneNumber}
            onChange={handlePhoneChange}
          />
        </div>

        <div className="input-group">
          <label>Password</label>
          <input
            name="password"
            type="password"
            value={form.password}
            onChange={handleChange}
            placeholder="Enter your password"
            autoComplete="new-password"
          />
        </div>

        <div className="input-group">
          <label>Confirm Password</label>
          <input
            name="confirmPassword"
            type="password"
            value={form.confirmPassword}
            onChange={handleChange}
            placeholder="Confirm your password"
            autoComplete="new-password"
          />
        </div>

        {error && <p className="auth-message auth-message-error">{error}</p>}
        {successMessage && <p className="auth-message auth-message-success">{successMessage}</p>}

        <button className="login-btn" type="submit" disabled={isSubmitting}>
          {isSubmitting ? "Creating account..." : "Sign Up"}
        </button>

        <p className="bottom-text">
          Already have an account? <Link to="/login">Login</Link>
        </p>
      </form>
    </div>
  );
}

export default Signup;
