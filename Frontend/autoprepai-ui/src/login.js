import { Link, useNavigate } from "react-router-dom";
import "./Auth.css";

import { Database } from "lucide-react";

function Login() {
  const navigate = useNavigate();
  return (
    <div className="auth-wrapper">
      <div className="auth-container">
        <div className="icon">
          <Database size={32} />
        </div>

        <h2>Login</h2>
        <p className="subtitle">Welcome back to AutoPrepAI</p>

        <div className="input-group">
          <label>Email or Username</label>
          <input placeholder="Enter your email or username" />
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
          <input type="password" placeholder="Enter your password" />
        </div>

        <button className="login-btn" onClick={() => navigate("/")}>
          Login
        </button>

        <p className="bottom-text">
          Don’t have an account? <Link to="/signup">Sign up</Link>
        </p>
      </div>
    </div>
  );
}

export default Login;
