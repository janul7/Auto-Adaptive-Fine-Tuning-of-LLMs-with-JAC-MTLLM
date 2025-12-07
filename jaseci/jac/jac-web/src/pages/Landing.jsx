"use client"

import { Link } from "react-router-dom"
import { useEffect } from "react"
import "../styles/landing.css"
import "../styles/showcase.css";
import Footer from "../components/Footer.jsx"
import "../styles/footer.css"
import Features from "../components/Features.jsx"
import Showcase from "../components/Showcase.jsx"
import ContactCTA from "../components/ContactCTA.jsx"
import ThemeToggle from "../components/EnhancedNavbar.jsx"
import DynamicShowcase from "../components/DynamicShowcase.jsx";
import EnhancedNavbar from "../components/EnhancedNavbar.jsx";
import "../styles/EnhancedNavbar.css";

export default function Landing() {
  useEffect(()=>{ document.body.dataset.page = "landing"; },[]);
  useEffect(() => {
    document.title = "JAC Web — Execute with Style"
  }, [])

  return (
    <div className="page" data-bg="landing">
      <div className="bg-dyn"></div>
      <div className="bg-image"></div>
      <div className="bg-vignette"></div>
      <div className="net-lines"></div>
      <div className="floating-orbs"></div>

    
      <EnhancedNavbar />
      <DynamicShowcase
        src="/media/intro.webm"
        mp4="/media/intro.mp4"
        poster="/media/intro-poster.jpg"
      />
      <section className="hero-wrap">
        <div className="hero-card elevate bounce-in">
          <h1 className="display">
            Execute your <span className="grad-text">JAC files</span> with style.
          </h1>
          <p className="lead">
            Upload, run, and monitor your code execution in a beautiful, modern interface designed for developers.
          </p>
          <div className="cta-row">
            <Link to="/upload" className="btn btn-hero">
              <span>📁</span> Upload & Run
            </Link>
            <Link to="/console" className="btn btn-ghost">
              <span>💻</span> Console
            </Link>
            <Link to="/app" className="btn btn-ghost">
              <span>🎮</span> Play RPG
            </Link>
          </div>
        </div>
        
      </section>
      <Features />
      <Showcase />
      <ContactCTA />
      <Footer />
    </div>
  )
}
