import React from "react"
import "./styles/theme.css";
import ReactDOM from "react-dom/client"
import { BrowserRouter, Routes, Route } from "react-router-dom"
import Home from "./pages/Home.jsx"
import Upload from "./pages/Upload.jsx"
import Console from "./pages/Console.jsx"
import Landing from "./pages/Landing.jsx"
import "./styles/global.css"
import "./styles/theme-toggle.css";
import About from "./pages/About.jsx";

// Set initial theme from localStorage or OS preference
const saved = localStorage.getItem("theme");
const prefersLight = window.matchMedia("(prefers-color-scheme: light)").matches;
document.documentElement.setAttribute("data-theme", saved || (prefersLight ? "light" : "dark"));

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/app" element={<Home />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/console" element={<Console />} />
        <Route path="/about" element={<About />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);