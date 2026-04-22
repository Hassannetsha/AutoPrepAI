import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Database } from "lucide-react";
import { loginUser, storeAuthToken } from "../../api/auth";
import "../../styles/Auth.css";

function Login() {
  const navigate = useNavigate();
  const [form, setForm] = useState({ email: "", password: "" });
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleChange = (event) => {
    const { name, value } = event.target;
    setForm((currentForm) => ({ ...currentForm, [name]: value }));
    setError("");
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!form.email.trim() || !form.password) {
      setError("Please enter your email and password.");
      return;
    }

    setIsSubmitting(true);
    setError("");

    try {
      const data = await loginUser({
        email: form.email.trim(),
        password: form.password,
      });

      storeAuthToken(data.access_token);
      navigate("/");
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

        <h2>Login</h2>
        <p className="subtitle">Welcome back to AutoPrepAI</p>

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
          <div className="password-row">
            <label>Password</label>

            <span className="forgot">
              <Link to="/forgot-password" className="forgot-link">
                Forgot password?
              </Link>
            </span>
          </div>
          <input
            name="password"
            type="password"
            value={form.password}
            onChange={handleChange}
            placeholder="Enter your password"
            autoComplete="current-password"
          />
        </div>

        {error && <p className="auth-message auth-message-error">{error}</p>}

        <button className="login-btn" type="submit" disabled={isSubmitting}>
          {isSubmitting ? "Logging in..." : "Login"}
        </button>

        <p className="bottom-text">
          Don&apos;t have an account? <Link to="/signup">Sign up</Link>
        </p>
      </form>
    </div>
  );
}

export default Login;
