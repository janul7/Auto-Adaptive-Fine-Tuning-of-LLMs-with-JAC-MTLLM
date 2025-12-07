"use client";
import { useEffect } from "react";
import "../styles/about.css";
import Footer from "../components/Footer.jsx";
import ContactCTA from "../components/ContactCTA.jsx";

export default function About(){
    useEffect(()=>{ document.body.dataset.page = "landing"; },[]);
  useEffect(()=>{ document.title = "About — JAC Web"; },[]);
  return (
    <div className="page" data-page="about">
      <div className="bg-dyn"></div>

      <header className="topnav glass">
        <div className="brand"><span className="logo">⚡</span> JARVIS</div>
        <nav>
          <a className="nav-link" href="/">Dashboard</a>
            <a className="nav-link" href="/app">Home</a>
            <a className="nav-link" href="/upload">Upload</a>
          <a className="nav-link" href="#contact">Contact</a>
        </nav>
      </header>

      <section className="about-hero">
        <h1 className="display grad-text">Build the future with Jaseci/JAC</h1>
        <p className="lead" style={{ textAlign: "center" }}>Auto-adaptive workflows, beautiful console UX, and rapid iteration for structured generation tasks.</p>
        <div className="cta-row">
          <a className="btn btn-hero" href="https://github.com/Hushan-10/Auto-Adaptive-Fine-tuning-for-Jac-MTLLM-using-RPG-Game-Generation" target="_blank" rel="noreferrer">View GitHub Repo</a>
        </div>
      </section>

      <section className="media-grid">
        <div className="card media">
          {/* YouTube sample; replace with your actual video ID */}
          <iframe className="yt" src="https://www.youtube.com/embed/MVJ6U6Yi8Bk"
            title="JAC Web Demo" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen />
        </div>
        <div className="card notes">
          <h3 className="fancy-heading">What's inside</h3>
            <ul className="feature-list">
              <li>
                  <span className="icon">⚡</span>
                  <div>
                    <h4>One-click Execution</h4>
                    <p>Upload & run JAC instantly with zero setup overhead.</p>
                  </div>
              </li>
              <li>
                <span className="icon">🖥️</span>
                <div>
                  <h4>Smart Console</h4>
                  <p>Filter noise with <em>“Show Only Important Output”</em> in real time.</p>
                </div>
              </li>
              <li>
                <span className="icon">🎮</span>
                <div>
                  <h4>RPG Validation</h4>
                  <p>Interactive demo to showcase auto-adaptive model behavior.</p>
                </div>
              </li>
              <li>
                <span className="icon">🌗</span>
                <div>
                  <h4>Dynamic Themes</h4>
                  <p>Switch seamlessly between light & dark with animated backdrops.</p>
                </div>
              </li>
            </ul>
        </div>

        <div className="card media">
          {/* Optional local clip; replace src with your asset path */}
          <video className="clip" controls src="/media/overview.mp4" />
        </div>
      </section>

      <section id="contact">
        <ContactCTA />
      </section>
      <Footer />
    </div>
  )
}
