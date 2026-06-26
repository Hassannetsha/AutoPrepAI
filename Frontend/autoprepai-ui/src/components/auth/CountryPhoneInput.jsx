import { useState, useRef, useEffect } from "react";

const FLAG_BASE = "https://flagcdn.com";

const COUNTRIES = [
  { code: "DZ", name: "Algeria", dial: "213" },
  { code: "AU", name: "Australia", dial: "61" },
  { code: "BH", name: "Bahrain", dial: "973" },
  { code: "BR", name: "Brazil", dial: "55" },
  { code: "CA", name: "Canada", dial: "1" },
  { code: "CN", name: "China", dial: "86" },
  { code: "EG", name: "Egypt", dial: "20" },
  { code: "FR", name: "France", dial: "33" },
  { code: "DE", name: "Germany", dial: "49" },
  { code: "IN", name: "India", dial: "91" },
  { code: "IT", name: "Italy", dial: "39" },
  { code: "JP", name: "Japan", dial: "81" },
  { code: "JO", name: "Jordan", dial: "962" },
  { code: "KR", name: "South Korea", dial: "82" },
  { code: "KW", name: "Kuwait", dial: "965" },
  { code: "LB", name: "Lebanon", dial: "961" },
  { code: "MY", name: "Malaysia", dial: "60" },
  { code: "MX", name: "Mexico", dial: "52" },
  { code: "MA", name: "Morocco", dial: "212" },
  { code: "NL", name: "Netherlands", dial: "31" },
  { code: "OM", name: "Oman", dial: "968" },
  { code: "QA", name: "Qatar", dial: "974" },
  { code: "SA", name: "Saudi Arabia", dial: "966" },
  { code: "SG", name: "Singapore", dial: "65" },
  { code: "ES", name: "Spain", dial: "34" },
  { code: "TN", name: "Tunisia", dial: "216" },
  { code: "TR", name: "Turkey", dial: "90" },
  { code: "AE", name: "United Arab Emirates", dial: "971" },
  { code: "GB", name: "United Kingdom", dial: "44" },
  { code: "US", name: "United States", dial: "1" },
];

function flagUrl(code) {
  return `${FLAG_BASE}/${code.toLowerCase()}.svg`;
}

function getFlagByDial(dial) {
  return COUNTRIES.find((c) => c.dial === dial);
}

function parsePhone(value) {
  if (!value) return { dial: "20", number: "" };
  const match = value.match(/^\+(\d+)(\d*)$/);
  if (!match) return { dial: "20", number: value.replace(/[^0-9]/g, "") };
  const full = match[1];
  for (let i = 4; i >= 1; i--) {
    const dial = full.slice(0, i);
    if (getFlagByDial(dial)) {
      return { dial, number: full.slice(i) };
    }
  }
  return { dial: full.slice(0, 3), number: full.slice(3) };
}

export default function CountryPhoneInput({ value, onChange, error }) {
  const { dial, number } = parsePhone(value);
  const [selectedDial, setSelectedDial] = useState(dial);
  const [localNumber, setLocalNumber] = useState(number);
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState("");
  const ref = useRef(null);
  const inputRef = useRef(null);

  const selected = getFlagByDial(selectedDial);

  useEffect(() => {
    const { dial: d, number: n } = parsePhone(value);
    setSelectedDial(d);
    setLocalNumber(n);
  }, [value]);

  useEffect(() => {
    function handleClick(e) {
      if (ref.current && !ref.current.contains(e.target)) {
        setOpen(false);
        setFilter("");
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const filtered = filter
    ? COUNTRIES.filter(
        (c) =>
          c.name.toLowerCase().includes(filter.toLowerCase()) ||
          c.dial.startsWith(filter) ||
          c.code.toLowerCase().includes(filter.toLowerCase())
      )
    : COUNTRIES;

  function selectCountry(country) {
    setSelectedDial(country.dial);
    setOpen(false);
    setFilter("");
    inputRef.current?.focus();
    const full = `+${country.dial}${localNumber}`;
    onChange(full);
  }

  function handleNumberChange(e) {
    const raw = e.target.value.replace(/[^0-9]/g, "");
    setLocalNumber(raw);
    const full = `+${selectedDial}${raw}`;
    onChange(full);
  }

  return (
    <div className={`phone-input-wrapper ${error ? "phone-input-error" : ""}`} ref={ref}>
      <div className="phone-country-select" onClick={() => setOpen(!open)}>
        {selected && <img className="phone-flag-img" src={flagUrl(selected.code)} alt={selected.code} />}
        <span className="phone-dial">+{selectedDial}</span>
        <span className={`phone-arrow ${open ? "phone-arrow-up" : ""}`}>▾</span>
      </div>

      {open && (
        <div className="phone-dropdown">
          <input
            className="phone-search"
            placeholder="Search country..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            autoFocus
          />
          <div className="phone-options">
            {filtered.map((c) => (
              <div
                key={c.code}
                className={`phone-option ${c.dial === selectedDial ? "phone-option-active" : ""}`}
                onClick={() => selectCountry(c)}
              >
                <img className="phone-flag-img" src={flagUrl(c.code)} alt={c.code} />
                <span className="phone-option-name">{c.name}</span>
                <span className="phone-option-dial">+{c.dial}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <input
        ref={inputRef}
        className="phone-number-input"
        type="tel"
        value={localNumber}
        onChange={handleNumberChange}
        placeholder="Enter phone number"
        autoComplete="tel-national"
      />
    </div>
  );
}
