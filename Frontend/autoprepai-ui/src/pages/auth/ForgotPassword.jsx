import { Link, useNavigate } from "react-router-dom";
import { Database } from "lucide-react";
import "../../styles/Auth.css";

function ForgotPassword() {
  const navigate = useNavigate();
  return (
    <div className="auth-wrapper">
      <div className="auth-container">
        <div className="icon">
          <Database size={32} />
        </div>

        <h2>Forgot Password?</h2>
        <p className="subtitle">
          Enter your email and we'll send you a reset link
        </p>

        <div className="input-group">
          <label>Email</label>
          <input type="email" placeholder="Enter your email" />
        </div>

        <button
          className="login-btn"
          onClick={() => navigate("/reset-password")}
        >
          Send Reset Link
        </button>

        <p className="bottom-text">
          Remember your password? <Link to="/login">Login</Link>
        </p>
      </div>
    </div>
  );
}

export default ForgotPassword;
