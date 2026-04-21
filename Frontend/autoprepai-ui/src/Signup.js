import { Link, useNavigate } from "react-router-dom";
import "./Auth.css";
import { Database } from "lucide-react";

function Signup() {
  const navigate = useNavigate();
  return (
    <div className="auth-wrapper">
      <div className="auth-container">
        <div className="icon">
          <Database size={32} />
        </div>

        <h2>Sign Up</h2>
        <p className="subtitle">Create your account</p>

        <div className="input-group">
          <label>Username</label>
          <input placeholder="Enter your username" />
        </div>

        <div className="input-group">
          <label>Email</label>
          <input placeholder="Enter your email" />
        </div>

        <div className="input-group">
          <label>Password</label>
          <input type="password" placeholder="Enter your password" />
        </div>

        <div className="input-group">
          <label>Confirm Password</label>
          <input type="password" placeholder="Confirm your password" />
        </div>

        <button className="login-btn" onClick={() => navigate("/verify-email")}>
          Sign Up
        </button>

        <p className="bottom-text">
          Already have an account? <Link to="/login">Login</Link>
        </p>
      </div>
    </div>
  );
}

export default Signup;
