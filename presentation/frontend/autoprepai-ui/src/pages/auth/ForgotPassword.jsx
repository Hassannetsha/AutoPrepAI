import { Link } from "react-router-dom";
import { useState } from "react";
import { Database } from "lucide-react";
import "../../styles/Auth.css";
import { forgotPassword } from "../../api/auth";

function ForgotPassword() {
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [message, setMessage] = useState(null);

  async function handleSubmit() {
    setError(null);
    setMessage(null);
    if (!email) return setError("Please enter your email.");
    setLoading(true);
    try {
      await forgotPassword({ email });
      setMessage("If that email is in our system, we have sent a reset link.");
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="auth-wrapper">
      <div className="auth-container">
        <div className="icon">
          <Database size={32} />
        </div>

        <h2>Forgot Password?</h2>
        <p className="subtitle">Enter your email and we'll send you a reset link</p>

        <div className="input-group">
          <label>Email</label>
          <input
            type="email"
            placeholder="Enter your email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </div>

        <button className="login-btn" onClick={handleSubmit} disabled={loading}>
          {loading ? "Sending..." : "Send Reset Link"}
        </button>

        {message && <p className="success-text">{message}</p>}
        {error && <p className="error-text">{error}</p>}

        <p className="bottom-text">
          Remember your password? <Link to="/login">Login</Link>
        </p>
      </div>
    </div>
  );
}

export default ForgotPassword;
