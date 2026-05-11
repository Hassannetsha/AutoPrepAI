import { useEffect, useState } from "react";
import { Database } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { getAuthToken, removeAuthToken } from "../../api/auth";

export default function AppHeader({ onLoginClick }) {
  const [isLoggedIn, setIsLoggedIn] = useState(!!getAuthToken());
  const navigate = useNavigate();

  useEffect(() => {
    const handleStorage = () => setIsLoggedIn(!!getAuthToken());
    window.addEventListener("storage", handleStorage);
    return () => window.removeEventListener("storage", handleStorage);
  }, []);

  function handleLogout() {
    removeAuthToken();
    setIsLoggedIn(false);
    navigate("/login");
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
