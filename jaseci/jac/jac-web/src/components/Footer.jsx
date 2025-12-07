"use client";
import { useEffect, useState } from "react";

export default function Footer(){
  const [showUp, setShowUp] = useState(false);
  useEffect(()=>{
    const onScroll = () => setShowUp(window.scrollY > 300);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  },[]);
  return (
    <footer className="site-footer">
      <div className="footer-wrap">
        <div className="brandline">
          <span className="logo">⚡</span>
          <strong>JARVIS</strong>
          <span className="tag">Lightning Fast • Precise Control • Built for Developers</span>
        </div>
        <div className="cols">
          <div className="col">
            <h4>Product</h4>
            <a href="#features">Features</a>
            <a href="#screens">Screenshots</a>
            <a href="/about">About</a>
          </div>
          <div className="col">
            <h4>Resources</h4>
            <a href="https://github.com/Hushan-10/Auto-Adaptive-Fine-tuning-for-Jac-MTLLM-using-RPG-Game-Generation" target="_blank" rel="noreferrer">GitHub</a>
            <a href="#contact">Contact</a>
          </div>
          <div className="col">
            <h4>Legal</h4>
            <a href="#">Terms</a>
            <a href="#">Privacy</a>
          </div>
        </div>
        <div className="copy">© {new Date().getFullYear()} JAC Web. All rights reserved.</div>
      </div>

      {showUp && (
        <button className="scroll-up" onClick={()=>window.scrollTo({top:0, behavior:"smooth"})} aria-label="Scroll to top">↑</button>
      )}
    </footer>
  );
}
