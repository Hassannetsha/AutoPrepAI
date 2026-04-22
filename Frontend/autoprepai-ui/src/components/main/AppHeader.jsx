import { Database } from "lucide-react";

export default function AppHeader({ onLoginClick }) {
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

      <button onClick={onLoginClick}>Login</button>
    </div>
  );
}
