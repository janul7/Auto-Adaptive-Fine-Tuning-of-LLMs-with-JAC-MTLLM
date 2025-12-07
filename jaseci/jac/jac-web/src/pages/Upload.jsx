"use client"

import { useState } from "react"
import { useNavigate } from "react-router-dom"
import "../styles/upload.css"
import Footer from "../components/Footer.jsx"
import "../styles/footer.css"
import { useEffect } from "react"

export default function Upload() {
  const [file, setFile] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const nav = useNavigate()

  async function onUpload() {
    if (!file) return alert("Choose a .jac file first")
    const fd = new FormData()
    fd.append("file", file)
    const r = await fetch("/api/upload", { method: "POST", body: fd })
    const j = await r.json().catch(() => ({}))
    if (!r.ok || j.ok === false) {
      alert(j.error || "Upload failed")
      return
    }
    const u = new URLSearchParams({ jac_path: j.jac_path, task_name: j.task_name })
    nav(`/console?${u.toString()}`)
  }
  useEffect(()=>{ document.body.dataset.page = "upload"; },[]);

  function handleDragOver(e) {
    e.preventDefault()
    setIsDragging(true)
  }

  function handleDragLeave(e) {
    e.preventDefault()
    setIsDragging(false)
  }

  function handleDrop(e) {
    e.preventDefault()
    setIsDragging(false)
    const files = e.dataTransfer.files
    if (files.length > 0) {
      setFile(files[0])
    }
  }

  return (
    <div className="page">
      <div className="bg-full">
        <div className="bg-gradient"></div>
      </div>
      <header className="topnav glass">
        <div className="brand"><span className="logo">⚡</span> JARVIS</div>
        <nav>
          <a className="nav-link" href="/">Dashboard</a>
          <a className="nav-link" href="/about">About</a>
          <a className="nav-link" href="/console">Console</a>
          <a className="nav-link" href="/app">Home</a>
        </nav>
      </header>
      <div className="container">
        <div className="card upload-card">
          <div className="card-header">
            <h1 className="h1">
              Select your <span className="highlight">JAC file</span> and watch it execute in real-time
            </h1>
          </div>

          <div
            className={`drop-zone ${isDragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="drop-icon">📄</div>
            <div className="drop-text">
              {file ? (
                <>
                  <strong>{file.name}</strong>
                  <br />
                  <small>Ready to upload</small>
                </>
              ) : (
                <>
                  <strong>Drop your JAC file here</strong>
                  <br />
                  <small>or click to browse</small>
                </>
              )}
            </div>
            <input
              type="file"
              accept=".jac"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="file-input"
            />
          </div>

          <div className="button-row">
            <button className="btn btn-primary" onClick={onUpload} disabled={!file}>
              ▶ Upload & Run
            </button>
            <button className="btn btn-secondary" onClick={() => nav("/app")}>
              ← Back to Home
            </button>
          </div>
        </div>
      </div>
      <Footer />
    </div>
  )
}
