
import { useEffect, useRef, useState } from "react";
import "../styles/dynamic-showcase.css";

export default function DynamicShowcaseFinal({
  webm = "/media/intro.webm",
  mp4 = "/media/intro.mp4",
  poster = "/media/intro-poster.jpg",
}) {
  // ticker terms
  const terms = ["JAC", "JARVIS", "Logs", "RPG", "Themes", "Deterministic", "Console UX"];
  const [idx, setIdx] = useState(0);

  // rotate ticker term
  useEffect(() => {
    const t = setInterval(() => setIdx((i) => (i + 1) % terms.length), 1600);
    return () => clearInterval(t);
  }, []);

  // subtle tilt on the video box
  const cardRef = useRef(null);
  useEffect(() => {
    const el = cardRef.current;
    if (!el) return;
    const onMove = (e) => {
      const r = el.getBoundingClientRect();
      const x = (e.clientX - r.left) / r.width - 0.5;
      const y = (e.clientY - r.top) / r.height - 0.5;
      el.style.setProperty("--rx", `${y * 3.5}deg`);
      el.style.setProperty("--ry", `${x * 5}deg`);
    };
    const onLeave = () => {
      el.style.setProperty("--rx", "0deg");
      el.style.setProperty("--ry", "0deg");
    };
    el.addEventListener("mousemove", onMove);
    el.addEventListener("mouseleave", onLeave);
    return () => {
      el.removeEventListener("mousemove", onMove);
      el.removeEventListener("mouseleave", onLeave);
    };
  }, []);

  return (
    <>
      <section className="hero-final" aria-label="JARVIS split hero">
        <div className="glow g1" />
        <div className="glow g2" />

        <div className="wrap">
          {/* LEFT CONTENT */}
          <div className="left">
            <h1 className="title">
              Execute <span className="grad">JAC</span> with
              <br /> precision, speed, and style.
            </h1>

            <p className="lead">
              Distilled Intelligence, Instant Execution-One Pipeline, Every Task. Cloud-Grade Quality Meets Edge-Fast Reflexes-Auto-Distilled, Fine-Tuned Local Models For Any Task.
            </p>

            <ul className="bullets">
              <li>⚡ One-click run & instant feedback</li>
              <li>🖥️ <em>Show Only Important Output</em> filter</li>
              <li>🎮 RPG demo for behavior validation</li>
              <li>🌗 Dynamic dark/light with animated backdrops</li>
            </ul>
          </div>

          {/* RIGHT VIDEO (no ribbon, no inner border) */}
          <div className="right">
            <div
              ref={cardRef}
              className="media-box"
              style={{
                transform:
                  "perspective(1200px) rotateX(var(--rx,0)) rotateY(var(--ry,0))",
                transition: "transform 120ms ease",
              }}
            >
              <video className="video" autoPlay muted loop playsInline poster={poster}>
                <source src={webm} type="video/webm" />
                <source src={mp4} type="video/mp4" />
              </video>
            </div>

            {/* chips BELOW the video */}
            <div className="chips-below">
              <span className="chip">Deterministic</span>
              <span className="chip">LLM-assisted</span>
              <span className="chip">Secure</span>
            </div>
          </div>
        </div>
      </section>

      {/* Dynamic keyword ticker under the section */}
      <section className="ticker" aria-label="keywords">
        <div className="ticker-inner">
          <span className="static">
            JAC • JARVIS • Logs • RPG • Themes • Deterministic •{" "}
          </span>
          <span className="static">
            JAC • JARVIS • Logs • RPG • Themes • Deterministic •{" "}
          </span>
          <span className="current">{terms[idx]}</span>
        </div>
      </section>
    </>
  );
}
