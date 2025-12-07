import { useEffect, useRef, useState } from "react";
import "../styles/features.css";

const FEATURES = [
  {
    icon: "🖥️",
    title: "Real-time Console",
    desc: "Streamed logs with smart highlighting, copy & clear controls.",
    cta: { label: "Open Console →", href: "/console" },
  },
  {
    icon: "📂",
    title: "One-click Upload & Run",
    desc: "Drop your JAC file and execute instantly with pretty output.",
    cta: { label: "Upload & Run →", href: "/upload" },
  },
  {
    icon: "🎮",
    title: "RPG Mode",
    desc: "Spin up the auto-adaptive RPG demo to test generative mechanics.",
    cta: { label: "Play RPG →", href: "/app" },
  },
  {
    icon: "🎯",
    title: "Precise Filters",
    desc: "Show only the important output when you want signal over noise.",
    cta: { label: "Learn More →", href: "/features" },
  },
];

export default function Features() {
  const trackRef = useRef(null);
  const [active, setActive] = useState(0);

  // update dot based on scroll position (mobile)
  useEffect(() => {
    const el = trackRef.current;
    if (!el) return;
    const onScroll = () => {
      const i = Math.round(el.scrollLeft / el.clientWidth);
      setActive(i);
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  const go = (dir) => {
    const el = trackRef.current;
    if (!el) return;
    const next = Math.max(0, Math.min(FEATURES.length - 1, active + dir));
    el.scrollTo({ left: next * el.clientWidth, behavior: "smooth" });
    setActive(next);
  };

  return (
    <section id="features"  className="features-section">
      <div className="section-wrap">
        <h2 className="section-title">What you get</h2>
        <p className="section-lead">Fast, focused tools for developers working with JAC.</p>

        {/* Desktop grid */}
        <div className="features-grid">
          {FEATURES.map((f, i) => (
            <article key={i} className="feat-card elevate">
              <div className="feat-icon" aria-hidden>{f.icon}</div>
              <h3 className="feat-title">{f.title}</h3>
              <p className="feat-desc">{f.desc}</p>
              <a className="feat-cta" href={f.cta.href}>{f.cta.label}</a>
            </article>
          ))}
        </div>

        {/* Mobile carousel */}
        <div className="features-carousel">
          <div className="carousel-track" ref={trackRef}>
            {FEATURES.map((f, i) => (
              <article key={`m-${i}`} className="feat-card elevate">
                <div className="feat-icon" aria-hidden>{f.icon}</div>
                <h3 className="feat-title">{f.title}</h3>
                <p className="feat-desc">{f.desc}</p>
                <a className="feat-cta" href={f.cta.href}>{f.cta.label}</a>
              </article>
            ))}
          </div>

          <div className="carousel-dots" role="tablist" aria-label="Features">
            {FEATURES.map((_, i) => (
              <button
                key={`dot-${i}`}
                className={`dot ${i === active ? "is-active" : ""}`}
                aria-label={`Slide ${i + 1}`}
                onClick={() => go(i - active)}
              />
            ))}
          </div>

          <div className="carousel-arrows">
            <button className="arrow" onClick={() => go(-1)} aria-label="Previous">‹</button>
            <button className="arrow" onClick={() => go(1)} aria-label="Next">›</button>
          </div>
        </div>
      </div>
    </section>
  );
}
