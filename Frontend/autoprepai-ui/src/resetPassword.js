import { useNavigate } from "react-router-dom";
import { Database } from "lucide-react";
import "./Auth.css";

function ResetPassword() {
  const navigate = useNavigate();
  return (
    <div className="auth-wrapper">
      <div className="auth-container">
        <div className="icon">
          <Database size={32} />
        </div>

        <h2>Reset Password</h2>
        <p className="subtitle">Enter your new password below</p>

        <div className="input-group">
          <label>New Password</label>
          <input type="password" placeholder="Enter new password" />
        </div>

        <div className="input-group">
          <label>Confirm New Password</label>
          <input type="password" placeholder="Confirm new password" />
        </div>

        <button className="login-btn" onClick={() => navigate("/login")}>
          Reset Password
        </button>

        <p className="bottom-text">Remember your password?</p>
      </div>
    </div>
  );
}

export default ResetPassword;
