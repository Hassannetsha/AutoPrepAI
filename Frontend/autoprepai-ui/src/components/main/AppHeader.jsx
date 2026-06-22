import { useEffect, useState } from "react";
import { Database } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { getAuthToken, logoutUser, removeAuthToken } from "../../api/auth";

export const LOGOUT_EVENT = "autoprepai_logout";

export function emitLogout() {
  removeAuthToken();
  window.dispatchEvent(new Event(LOGOUT_EVENT));
  // best-effort backend call, don't block the UI
  logoutUser().catch(() => {});
}

export default function AppHeader({ onLoginClick }) {
  const [isLoggedIn, setIsLoggedIn] = useState(!!getAuthToken());
  const navigate = useNavigate();

  useEffect(() => {
    const handleStorage = () => setIsLoggedIn(!!getAuthToken());
    const handleLogoutEvent = () => {
      setIsLoggedIn(false);
      // navigate("/login", { replace: true });
    };
    window.addEventListener("storage", handleStorage);
    window.addEventListener(LOGOUT_EVENT, handleLogoutEvent);
    return () => {
      window.removeEventListener("storage", handleStorage);
      window.removeEventListener(LOGOUT_EVENT, handleLogoutEvent);
    };
  }, [navigate]);

  function handleLogout() {
    emitLogout();
  }

  return (
    <div className="header">
      <div className="header-brand">
        <div className="header-icon">
          <Database size={22} />
        </div>
        <div>
          <h2>AutoPrepAI</h2>
          <p>AI-Powered Data Cleaning & Preparation</p>
        </div>
      </div>

      {!isLoggedIn ? (
        <button onClick={() => navigate("/login")}>Login</button>
      ) : (
        <button onClick={handleLogout}>Logout</button>
      )}
    </div>
  );
}
