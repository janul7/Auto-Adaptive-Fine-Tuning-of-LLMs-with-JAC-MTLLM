"use client"

import { useEffect, useRef, useState } from "react"
import { Link, useLocation } from "react-router-dom"
import "../styles/console.css"
import "../styles/footer.css"
import Footer from "../components/Footer.jsx"



  

function wsUrlWithParams(jac_path, task_name) {
  const proto = location.protocol === "https:" ? "wss" : "ws"
  const base = `${proto}://${location.host}`
  const qs = new URLSearchParams({ jac_path, task_name }).toString()
  return `${base}/ws/run?${qs}`
}

function isSchemaJson(line) {
  return /{\s*"schema_object_wrapper"\s*:\s*"[^"]+"\s*}/.test(line)
}
function isDetected(line) {
  return /personality\s+detected\s+for/i.test(line)
}
function isUserEcho(line) {
  return /^\s*>\s*/.test(line)
}

export default function Console() {
  const [exited, setExited] = useState(false)
  const stoppedRef = useRef(false) 
  const location = useLocation()
  const params = new URLSearchParams(location.search)
  const jac_path = params.get("jac_path") || ""
  const task_name = params.get("task_name") || "task"

  const [lines, setLines] = useState([])
  const [onlyImportant, setOnlyImportant] = useState(true)
  const [inp, setInp] = useState("")
  const [connected, setConnected] = useState(false)
  const wsRef = useRef(null)
  const terminalRef = useRef(null)


  useEffect(()=>{ document.body.dataset.page = "landing"; },[]);
  useEffect(() => {
    let stopped = false

    function connect() {
      if (stopped || stoppedRef.current) return
      const s = new WebSocket(wsUrlWithParams(jac_path, task_name))
      wsRef.current = s

      s.onopen = () => setConnected(true)
      s.onclose = () => {
        setConnected(false)
        if (!stopped) setTimeout(connect, 500)
      }
      s.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data)
          if (msg.type === "stdout" || msg.type === "event") {
            setLines((p) => [...p, msg.data ?? ""])
          }
        } catch {}
      }
    }

    connect()
    return () => {
      stopped = true
      try {
        wsRef.current?.close()
      } catch {}
    }
  }, [jac_path, task_name])

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight
    }
  }, [lines])

  function send() {
    const val = inp.trim()
    if (!val) return
    setLines((p) => [...p, `> ${val}`])
    wsRef.current?.send(JSON.stringify({ type: "stdin", data: val }))
    setInp("")
  }
  function exitApp() {
    // ✅ stop WS reconnect loop and close socket immediately
    stoppedRef.current = true
    try { wsRef.current?.close() } catch {}

    // Stop backend (ignore network errors)
    fetch("/api/shutdown", { method: "POST" }).catch(() => {})

    // Try to close the tab (works if opened by script/electron/tauri)
    window.open("", "_self"); window.close()

    // Show overlay if the tab didn’t close
    setExited(true)
  }
  const list = onlyImportant ? lines.filter((ln) => isUserEcho(ln) || isSchemaJson(ln) || isDetected(ln)) : lines

  return (
    <div className="page console-page">
      <div className="bg-full">
        <div className="bg-gradient"></div>
      </div>
      <header className="topnav glass">
        <div className="brand"><span className="logo">⚡</span> JARVIS</div>
        <nav>
          <a className="nav-link" href="/">Dashboard</a>
          <a className="nav-link" href="/about">About</a>
          <a className="nav-link" href="/app">Home</a>
          <a className="nav-link" href="/upload">Upload</a>
        </nav>
      </header>
      <div className="container">
        {!exited && (
        <div className="console-card">
          <div className="console-header">
            <div className="console-title">
              <div className="status-indicator">
                <div className={`status-dot ${connected ? "connected" : "disconnected"}`}></div>
                <span>{connected ? "Connected" : "Connecting..."}</span>
              </div>
              <h1 className="console-h1">Console — {task_name}</h1>
            </div>

            <label className="filter-toggle">
              <input type="checkbox" checked={onlyImportant} onChange={(e) => setOnlyImportant(e.target.checked)} />
              <span>Show only important output</span>
            </label>
          </div>

          <div className="terminal-wrapper">
            <div className="terminal-header">
              <div className="terminal-dots">
                <span className="dot red"></span>
                <span className="dot yellow"></span>
                <span className="dot green"></span>
              </div>
              <div className="terminal-title">[started] jac run {task_name}</div>
            </div>

            <pre className="terminal" ref={terminalRef}>
              {list.map((l, i) => (
                <div
                  key={i}
                  className={isUserEcho(l) ? "user-input" : isSchemaJson(l) ? "schema-output" : "normal-output"}
                >
                  {l}
                </div>
              ))}
            </pre>
          </div>

          <div className="input-row">
            <input
              className="terminal-input"
              placeholder="If this program expects input (e.g., cal.jac), type it here and press Enter. "
              value={inp}
              onChange={(e) => setInp(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault()
                  send()
                }
              }}
            />

            <button className="btn btn-send" onClick={send}>
              🔼 Send
            </button>
          </div>

          <div className="control-row">
            <button className="btn btn-danger" onClick={() => wsRef.current?.close()}>
              ■ Stop Task
            </button>
            <Link className="btn btn-secondary" to="/app">
              ← Go to Home
            </Link>
            <button className="btn btn-danger" onClick={exitApp}>
              ❌ Exit App
            </button>
          </div>
        </div>
        )}
      
      {exited && (
          <div className="overlay-exit">
            <h2>👋 👋 EXIT from JAC</h2>
            <p>You can now safely close this tab</p>
          </div>
        )}
    </div>
    <Footer/>
  </div>
  )
}