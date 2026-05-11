import { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Database } from "lucide-react";
import "../../styles/Auth.css";
import { resetPassword } from "../../api/auth";

function ResetPassword() {
  const navigate = useNavigate();
  const { search } = useLocation();
  const params = new URLSearchParams(search);
  const tokenFromQuery = params.get("token") || "";

  const [token, setToken] = useState(tokenFromQuery);
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleSubmit() {
    setError(null);
    if (!token) return setError("Missing reset token. Check your email link.");
    if (!newPassword) return setError("Please enter a new password.");
    if (newPassword !== confirmPassword) return setError("Passwords do not match.");

    setLoading(true);
    try {
      await resetPassword({ token, newPassword });
      navigate("/login");
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

        <h2>Reset Password</h2>
        <p className="subtitle">Enter your new password below</p>

        {!tokenFromQuery && (
          <div className="input-group">
            <label>Reset Token</label>
            <input
              type="text"
              placeholder="Paste reset token"
              value={token}
              onChange={(e) => setToken(e.target.value)}
            />
          </div>
        )}

        <div className="input-group">
          <label>New Password</label>
          <input
            type="password"
            placeholder="Enter new password"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
          />
        </div>

        <div className="input-group">
          <label>Confirm New Password</label>
          <input
            type="password"
            placeholder="Confirm new password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
          />
        </div>

        <button className="login-btn" onClick={handleSubmit} disabled={loading}>
          {loading ? "Resetting..." : "Reset Password"}
        </button>

        {error && <p className="error-text">{error}</p>}

        <p className="bottom-text">Remember your password?</p>
      </div>
    </div>
  );
}

export default ResetPassword;
