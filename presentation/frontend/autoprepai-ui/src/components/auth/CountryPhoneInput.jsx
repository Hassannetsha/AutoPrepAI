import { useState, useEffect } from "react";

// const EGYPT = { code: "EG", name: "Egypt", dial: "20" };

function parsePhone(value) {
  if (!value) return { dial: "20", number: "" };
  const match = value.match(/^\+20(\d*)$/);
  return { dial: "20", number: match ? match[1] : value.replace(/[^0-9]/g, "") };
}

export default function CountryPhoneInput({ value, onChange, error }) {
  const { number } = parsePhone(value);
  const [localNumber, setLocalNumber] = useState(number);

  useEffect(() => {
    const { number: n } = parsePhone(value);
    setLocalNumber(n);
  }, [value]);

  function handleNumberChange(e) {
    const raw = e.target.value.replace(/[^0-9]/g, "").slice(0, 10);
    setLocalNumber(raw);
    onChange(`+20${raw}`);
  }

  return (
    <div className={`phone-input-wrapper ${error ? "phone-input-error" : ""}`}>
      <div className="phone-country-select" style={{ cursor: "default" }}>
        <span className="phone-dial">+20</span>
      </div>
      <input
        className="phone-number-input"
        type="tel"
        value={localNumber}
        onChange={handleNumberChange}
        placeholder="10XXXXXXXX"
        autoComplete="tel-national"
      />
    </div>
  );
}
