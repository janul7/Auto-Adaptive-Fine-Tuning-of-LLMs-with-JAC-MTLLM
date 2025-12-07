import { useEffect, useState } from "react";
import "../styles/theme-toggle.css";

export default function ThemeToggle() {
  const getInitial = () => {
    const saved = localStorage.getItem("theme");
    if (saved) return saved;
    return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  };

  const [mode, setMode] = useState(getInitial);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", mode);
    localStorage.setItem("theme", mode);
  }, [mode]);

  const next = mode === "dark" ? "light" : "dark";

  return (
    <button
      className="tt"
      type="button"
      aria-label={`Switch to ${next} mode`}
      onClick={() => setMode(next)}
    >
      <span className={`tt-switch ${mode}`}>
        <span className="tt-thumb" aria-hidden>
          {mode === "dark" ? "🌙🌙" : "☀️☀️"}
        </span>
      </span>
      <span className="tt-label">{mode === "dark" ? "Dark" : "Light"}</span>
    </button>
  );
}
