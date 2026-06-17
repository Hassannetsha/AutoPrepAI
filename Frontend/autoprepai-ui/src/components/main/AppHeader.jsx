import { useEffect, useState } from "react";
import { Database } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { getAuthToken, logoutUser } from "../../api/auth";

export const LOGOUT_EVENT = "autoprepai_logout";

export async function emitLogout() {
  try {
    await logoutUser();
  } catch {
    // best-effort — clear local state regardless
  }
  localStorage.removeItem("autoprepai_access_token");
  window.dispatchEvent(new Event(LOGOUT_EVENT));
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
