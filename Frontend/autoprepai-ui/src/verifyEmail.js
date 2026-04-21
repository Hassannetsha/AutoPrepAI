import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { MailCheck } from "lucide-react";
import "./Auth.css";

function VerifyEmail() {
  const [code, setCode] = useState(["", "", "", "", "", ""]);
  const [timer, setTimer] = useState(60);
  const [canResend, setCanResend] = useState(false);
  const [error, setError] = useState("");
  const inputs = useRef([]);
  const navigate = useNavigate();

  // ✅ Countdown timer
  useEffect(() => {
    if (timer === 0) {
      setCanResend(true);
      return;
    }
    const interval = setInterval(() => {
      setTimer((prev) => prev - 1);
    }, 1000);
    return () => clearInterval(interval);
  }, [timer]);

  // ✅ Handle each digit input
  const handleChange = (value, index) => {
    if (!/^\d*$/.test(value)) return; // numbers only

    const newCode = [...code];
    newCode[index] = value.slice(-1); // only last digit
    setCode(newCode);
    setError("");

    // Auto move to next input
    if (value && index < 5) {
      inputs.current[index + 1]?.focus();
    }
  };

  // ✅ Handle backspace
  const handleKeyDown = (e, index) => {
    if (e.key === "Backspace" && !code[index] && index > 0) {
      inputs.current[index - 1]?.focus();
    }
  };

  // ✅ Resend code
  const handleResend = () => {
    if (!canResend) return;
    setTimer(60);
    setCanResend(false);
    setCode(["", "", "", "", "", ""]);
    setError("");
    inputs.current[0]?.focus();
    console.log("Resend code triggered");
  };

  // ✅ Submit
  const handleVerify = () => {
    const fullCode = code.join("");
    if (fullCode.length < 6) {
      setError("Please enter the full 6-digit code.");
      return;
    }
    console.log("Verifying code:", fullCode);
    navigate("/login"); // go to login after verify
  };

  return (
    <div className="auth-wrapper">
      <div className="auth-container">
        <div className="icon">
          <MailCheck size={32} />
        </div>

        <h2>Verify Your Email</h2>
        <p className="subtitle">
          We sent a 6-digit code to your email. <br />
          Enter it below to continue.
        </p>

        {/* ✅ 6 digit boxes */}
        <div className="code-inputs">
          {code.map((digit, index) => (
            <input
              key={index}
              ref={(el) => (inputs.current[index] = el)}
              type="text"
              inputMode="numeric"
              maxLength={1}
              value={digit}
              onChange={(e) => handleChange(e.target.value, index)}
              onKeyDown={(e) => handleKeyDown(e, index)}
              className="code-box"
            />
          ))}
        </div>

        {error && <p className="uploadError">{error}</p>}

        {/* ✅ Timer */}
        <p className="timer-text">
          {canResend ? (
            <span className="resend-link" onClick={handleResend}>
              Resend Code
            </span>
          ) : (
            <>
              Resend code in <strong>{timer}s</strong>
            </>
          )}
        </p>

        <button className="login-btn" onClick={handleVerify}>
          Verify Email
        </button>

        <p className="bottom-text">
          Wrong email?{" "}
          <span className="resend-link" onClick={() => navigate("/signup")}>
            Go back
          </span>
        </p>
      </div>
    </div>
  );
}

export default VerifyEmail;
