// import ThemeToggle from "./ThemeToggle";
// export default function EnhancedNavbar(){
//   return (
//     <header className="topnav glass">
//       <div className="brand"><span className="logo">⚡</span> JARVIS</div>
//       <nav>
//         <a className="nav-link" href="/app">Home</a>
//         <a className="nav-link" href="/upload">Upload</a>
//         <a className="nav-link" href="/console">Console</a>
//         <a className="nav-link" href="/#features">Features</a>
//         <a className="nav-link" href="/#screens">Screens</a>
//         <a className="nav-link" href="/about">About</a>
//       </nav>
//       <ThemeToggle />
//     </header>
//   );
// }


import { useState } from "react";
import ThemeToggle from "./ThemeToggle";

export default function EnhancedNavbar() {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <header className="topnav glass">
      <div className="brand">
        <span className="logo">⚡</span> JARVIS
      </div>

      {/* Hamburger for mobile */}
      <button
        className="hamburger"
        onClick={() => setMenuOpen(!menuOpen)}
        aria-label="Toggle navigation"
      >
        ☰
      </button>

      {/* Sidebar nav */}
      <nav className={`nav-links ${menuOpen ? "open" : ""}`}>
        <a className="nav-link" href="/app">Home</a>
        <a className="nav-link" href="/upload">Upload</a>
        <a className="nav-link" href="/console">Console</a>
        <a className="nav-link" href="/#features">Features</a>
        <a className="nav-link" href="/#screens">Screens</a>
        <a className="nav-link" href="/about">About</a>
      </nav>

      <ThemeToggle />
    </header>
  );
}
