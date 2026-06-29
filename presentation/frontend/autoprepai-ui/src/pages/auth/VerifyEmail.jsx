import { useEffect, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { MailCheck } from "lucide-react";
import { resendVerificationEmail, verifyEmailToken } from "../../api/auth";
import "../../styles/Auth.css";

function VerifyEmail() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const token = searchParams.get("token");
  const initialEmail = searchParams.get("email") || "";

  const [email, setEmail] = useState(initialEmail);
  const [status, setStatus] = useState(token ? "loading" : "idle");
  const [message, setMessage] = useState(
    token ? "Verifying your email..." : "Enter your email if you need another verification link."
  );
  const [error, setError] = useState("");
  const [isResending, setIsResending] = useState(false);

  useEffect(() => {
    if (!token) return;

    let isMounted = true;

    async function verify() {
      setStatus("loading");
      setError("");

      try {
        const data = await verifyEmailToken(token);
        if (!isMounted) return;
        setStatus("success");
        setMessage(data.message || "Email verified successfully! You can now log in.");
      } catch (err) {
        if (!isMounted) return;
        setStatus("error");
        setError(err.message);
      }
    }

    verify();

    return () => {
      isMounted = false;
    };
  }, [token]);

  const handleResend = async (event) => {
    event.preventDefault();

    if (!email.trim()) {
      setError("Please enter your email address first.");
      return;
    }

    setIsResending(true);
    setError("");
    setMessage("");

    try {
      const data = await resendVerificationEmail(email.trim());
      setStatus("idle");
      setMessage(data.message || "A new verification link has been sent to your email.");
    } catch (err) {
      setError(err.message);
    } finally {
      setIsResending(false);
    }
  };

  return (
    <div className="auth-wrapper">
      <form className="auth-container" onSubmit={handleResend}>
        <div className="icon">
          <MailCheck size={32} />
        </div>

        <h2>Verify Your Email</h2>
        <p className="subtitle">
          {token
            ? "We are confirming your verification link."
            : "Check your inbox for the verification link we sent."}
        </p>

        {message && (
          <p className={`auth-message ${status === "success" ? "auth-message-success" : "auth-message-info"}`}>
            {message}
          </p>
        )}
        {error && <p className="auth-message auth-message-error">{error}</p>}

        {status === "success" ? (
          <button className="login-btn" type="button" onClick={() => navigate("/login")}> 
            Go to Login
          </button>
        ) : (
          <>
            <div className="input-group">
              <label>Email</label>
              <input
                name="email"
                type="email"
                value={email}
                onChange={(event) => {
                  setEmail(event.target.value);
                  setError("");
                }}
                placeholder="Enter your email"
                autoComplete="email"
              />
            </div>

            <button className="login-btn" type="submit" disabled={isResending || status === "loading"}>
              {isResending ? "Sending..." : "Resend Verification Email"}
            </button>
          </>
        )}

        <p className="bottom-text">
          Already verified? <Link to="/login">Login</Link>
        </p>
      </form>
    </div>
  );
}

export default VerifyEmail;
