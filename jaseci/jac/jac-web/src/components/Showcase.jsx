"use client";
// src/components/Showcase.jsx
import homeImg    from "../assets/images/home-bg.png";
import consoleImg from "../assets/images/console-bg.png";
import uploadImg  from "../assets/images/upload-bg.png";
import NewMap     from "../assets/images/new-map.png";
import "../styles/showcase.css";

export default function Showcase() {
  const items = [
    { src: homeImg,    caption: "RPG Game" },
    { src: consoleImg, caption: "Console-Live Logs" },
    { src: uploadImg,  caption: "Upload & Run" },
    { src: NewMap,     caption: "Check Maps" },
  ];

  return (
    <section id="screens" className="showcase-section">
      <div className="section-wrap">
        <h2 className="section-title">See it in Action</h2>
        <p className="section-lead">A few screenshots from JARVIS.</p>

        <div className="shots">
          {items.map((it, i) => (
            <figure key={i} className="shot">
              <div className="img-box contain"> {/* contain = no cropping */}
                <img loading="lazy" src={it.src} alt={it.caption} />
                <div className="vignette" aria-hidden="true" />
              </div>
              <figcaption>{it.caption}</figcaption>
            </figure>
          ))}
        </div>
      </div>
    </section>
  );
}
