# server/main.py

import os, re, sys, json, time, queue, shutil, asyncio, pathlib, threading, subprocess, logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from collections import deque
import io
# --- Graceful app shutdown (stops child tasks, then kills uvicorn) ---
import signal, threading

def stop_all_sessions():
    try:
        for s in list(SESSIONS.values()):
            try:
                s.stop()
            except Exception:
                pass
        SESSIONS.clear()
    except NameError:
        pass

USER_TEXT_RE = re.compile(r"text'\s*:\s*'(.+?)'", re.DOTALL)

logging.basicConfig(level=logging.INFO, format="%(message)s")
# Best-effort: ensure our own stdout/stderr are UTF-8 (Python 3.7+)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
logger = logging.getLogger("jac_exec_server")

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent  # <-- up one level from /server
EXAMPLES_DIR = pathlib.Path(
    os.getenv("JAC_EXAMPLES_DIR", ROOT / "examples")
).resolve()
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

JAC_BIN = os.getenv("JAC_BIN") or shutil.which("jac") or "jac"

BASE_ENV = os.environ.copy()
if sys.platform.startswith("linux"):
    BASE_ENV.setdefault("SDL_VIDEODRIVER", "x11")
elif sys.platform == "win32":
    BASE_ENV.pop("SDL_VIDEODRIVER", None)   # <- do NOT force windib
    BASE_ENV.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    # Force UTF-8 for child Python process on Windows
    BASE_ENV.setdefault("PYTHONUTF8", "1")
    BASE_ENV.setdefault("PYTHONIOENCODING", "utf-8")
else:
    # Non-Windows fallback: still good to force UTF-8
    BASE_ENV.setdefault("PYTHONUTF8", "1")
    BASE_ENV.setdefault("PYTHONIOENCODING", "utf-8")

SYSTEM_PROMPT = (
    "This is a task you must complete by returning only the output.\n"
    "Do not include explanations, code, or extra text—only the result.\n"
)

# -----------------
# Utils
# -----------------
def safe_task_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", name or "").strip("_") or "task"


def task_dir_for(task: str) -> pathlib.Path:
    d = EXAMPLES_DIR / safe_task_name(task)
    d.mkdir(parents=True, exist_ok=True)   # ✅ only the task folder
    return d

def materialize_task_file(task: str, src: pathlib.Path) -> pathlib.Path:
    """
    Ensure a copy of `src` exists under jac/examples/<task>/, return that path.
    If src is already there, nothing to do.
    """
    src = src.resolve()
    dest_dir = task_dir_for(task)
    dest = dest_dir / src.name
    if dest.resolve() != src:
        dest.write_bytes(src.read_bytes())         # copy (use shutil.copy2 if you prefer metadata)
    return dest
# -----------------
# Utils
# -----------------

def append_llm_io(task_dir: pathlib.Path, system: str, inp: str, out: Any):
    """Always write flat log file beside the task file."""
    log_path = task_dir / "llm_io_log.jsonl"
    entry = {"system": system, "input": inp, "output": out}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


# -----------------
# FastAPI
# -----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:5173"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# -----------------
# Session
# -----------------
class JacSession:
    def __init__(self, task: str, jac_path: pathlib.Path, run_cwd: Optional[pathlib.Path] = None):
        self.task = safe_task_name(task)
        self.task_dir = task_dir_for(task)
        self.jac_path = jac_path.resolve()
        self.run_cwd = run_cwd or self.task_dir
        self.proc: Optional[subprocess.Popen] = None
        self.stdout_q: "queue.Queue[str]" = queue.Queue()
        self.stop_flag = threading.Event()
        # >>> multi-input safety: queue all pending inputs in order
        self.pending_inp = deque()                         # was Optional[str]
        # >>> single writer thread and queue to serialize stdin writes
        self._in_q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._writer_thread: Optional[threading.Thread] = None
        # <<<
        self.next_input_for_log: Optional[str] = None
        self._out_buf: list[str] = []   # collect lines until we see the final JSON
        self._last_logged_inp: Optional[str] = None     # dedupe guard
        self._pending_log_inp: Optional[str] = None     # staging for RPG-detected input

    def spawn(self):
        cmd = [JAC_BIN, "run", str(self.jac_path)]
        self.proc = subprocess.Popen(
            
            cmd,
            cwd=str(self.run_cwd),
            env=BASE_ENV,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            encoding="utf-8",           # <<< ensure UTF-8 pipes
            errors="replace",           # <<< never crash on odd glyphs

            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform=="win32" else 0,
        )
        threading.Thread(target=self._reader, daemon=True).start()
        # >>> start the writer thread
        self._writer_thread = threading.Thread(target=self._writer, daemon=True)
        self._writer_thread.start()
        # <<<

    def _reader(self):
        try:
            for line in self.proc.stdout:
                self.stdout_q.put(line)
        except Exception:
            pass
        finally:
            self.stdout_q.put("__EOF__")

    # >>> new: the single writer that owns stdin
    def _writer(self):
        while True:
            item = self._in_q.get()
            if item is None:
                break
            data = (item or "")
            # normalize and avoid empty CRLF spam
            data = str(data).replace("\r\n", "\n").rstrip("\n")
            # write only if process is alive and pipe is valid
            if not data and (self.proc is None or self.proc.poll() is not None):
                continue
            try:
                if self.proc and self.proc.stdin and self.proc.poll() is None:
                    self.proc.stdin.write(data + "\n")
                    self.proc.stdin.flush()
                else:
                    break
            except (OSError, ValueError, io.UnsupportedOperation):
                # stdin invalid/closed; end writer
                break
    # <<<


    async def pump(self, websocket: Optional[WebSocket] = None):
        while not self.stop_flag.is_set():
            line = await asyncio.to_thread(self.stdout_q.get)
            if line == "__EOF__":
                break
            stripped = line.rstrip("\n")

            # Mirror every process line to the server terminal only (UTF-8 safe)
            try:
            #    logger.info(f"[{self.task}] {stripped}")
                print(stripped)
            except Exception:
                # Fallback if terminal still isn't UTF-8
                # logger.info(f"[{self.task}] {stripped.encode('utf-8','replace').decode('utf-8')}")
                print(stripped.encode("utf-8", "replace").decode("utf-8"))          
        # always echo to UI (but don't crash if the socket is gone)
            if websocket:
               try:
                   await websocket.send_text(json.dumps({"type": "stdout", "data": stripped}))
               except Exception:
                   # Detach from WS; keep the subprocess alive
                   websocket = None

        # Detect RPG "user" content from printed messages
            if "'messages':" in stripped and "'role': 'user'" in stripped:
                m = USER_TEXT_RE.search(stripped)
                if m:
                    self.next_input_for_log = m.group(1).replace("\\n", "\n").strip()
                    self._last_logged_inp = None            # ← new: new input detected, allow one log
            # don't log yet; wait for final JSON
                continue

        # Try to parse this line as JSON (treat as a potential final output)
            parsed = None
            try:
                parsed = json.loads(stripped)
            except Exception:
                pass

            # Log ONLY mtllm plugin outputs that arrive as *JSON strings* (e.g. "{\"k\":\"v\"}")
            if isinstance(parsed, str):
             # Choose the input to pair with this output (RPG-detected first, else Console stdin)
                 if self.next_input_for_log is not None:
                     inp = self.next_input_for_log
                     self.next_input_for_log = None
                 elif self.pending_inp:
                     inp = self.pending_inp.popleft()
                 else:
                     inp = None  # no input -> don't log
                                 # Keep only strings that *look like* JSON payloads: "{...}" or "[...]"
                 payload = parsed.strip()
                 if inp is not None and (payload.startswith("{") or payload.startswith("[")):
                    if self._last_logged_inp == inp:
                         continue            # already logged for this input; ignore duplicate JSONs
                    # IMPORTANT: write the *inner* JSON string (parsed) so the log shows:
                    # "output": "{\"k\":\"v\"}"
                    append_llm_io(self.task_dir, SYSTEM_PROMPT, inp, payload)
                    self._last_logged_inp = inp
            elif isinstance(parsed, dict):
                # Skip manual/converted objects; we only want the plugin’s JSON-string outputs
                continue



    async def write_stdin(self, data: str):
        # >>> queue the input (pairing) and send to writer thread
        data = "" if data is None else str(data)
        self.pending_inp.append(data)
        self._last_logged_inp = None                # ← new: allow one log for this new input
        # non-blocking enqueue; writer thread will handle the actual pipe write
        self._in_q.put_nowait(data)
        # <<<

    def stop(self):
        self.stop_flag.set()
        # >>> stop writer
        try:
            self._in_q.put_nowait(None)
        except Exception:
            pass
        # <<<
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
        except:
            pass


SESSIONS: Dict[str, JacSession] = {}

# -----------------
# Endpoints
# -----------------
@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".jac"):
        raise HTTPException(400, "Only .jac files allowed")

    task = pathlib.Path(file.filename).stem        # task name from base
    # Save uploaded file into jac/examples/<task>/<filename>
    dest_dir = task_dir_for(task)
    dest_path = dest_dir / file.filename
    dest_path.write_bytes(await file.read())

    # Restart session
    old = SESSIONS.pop(task, None)
    if old: old.stop()

    sess = JacSession(task, dest_path)            # run_cwd -> <examples>/<task>
    sess.spawn()
    SESSIONS[task] = sess
    asyncio.create_task(sess.pump())
    return {"ok": True, "task_name": task, "jac_path": str(dest_path)}


def find_rpg_main() -> pathlib.Path:
    # search under common roots
    roots = [
        EXAMPLES_DIR / "rpg_game",
        HERE.parent / "examples" / "rpg_game",
        HERE / "examples" / "rpg_game",
    ]

    for root in roots:
        if root.exists():
            cands = list(root.glob("**/main.jac"))
            if cands:
                return cands[0].resolve()

    raise FileNotFoundError("No main.jac found under rpg_game")


@app.post("/api/rpg/run")
async def api_rpg_run():
    try:
        rpg = find_rpg_main()
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    old = SESSIONS.pop("rpg_game", None)
    if old:
        old.stop()

    # >>> set log dir to the directory of main.jac
    sess = JacSession("rpg_game", rpg, run_cwd=rpg.parent)
    sess.task_dir = rpg.parent   # override: log file will be created here
    sess.spawn()   # 🚀 actually start the process!
    # <<<

    (sess.task_dir / "llm_io_log.jsonl").touch(exist_ok=True)

    SESSIONS["rpg_game"] = sess
    asyncio.create_task(sess.pump())
    return {"ok": True, "task": "rpg_game", "jac_path": str(rpg)}


# -----------------
# WebSockets
# -----------------
@app.websocket("/ws/run")
async def ws_run(ws: WebSocket):
    await ws.accept()
    from urllib.parse import parse_qs, unquote_plus
    q = parse_qs(ws.scope["query_string"].decode())
    jac_path_q = unquote_plus(q.get("jac_path", [""])[0])  # may be filename or absolute path
    task = safe_task_name(q.get("task_name", ["task"])[0])

    # Resolve source file
    src = pathlib.Path(jac_path_q)
    if not src.exists():
        # allow relative lookup inside examples/<task> or examples/
        cand = task_dir_for(task) / pathlib.Path(jac_path_q).name
        if cand.exists():
            src = cand
        else:
            cand2 = EXAMPLES_DIR / jac_path_q
            if cand2.exists():
                src = cand2
    if not src.exists():
        await ws.close()
        return

    # Always ensure a copy under jac/examples/<task>
    dest = materialize_task_file(task, src)

    # one active session per task
    prev = SESSIONS.pop(task, None)
    if prev: prev.stop()

    sess = JacSession(task, dest, run_cwd=dest.parent)     # run from examples/<task>
    sess.spawn()
    SESSIONS[task] = sess
    asyncio.create_task(sess.pump(ws))

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
                if msg.get("type") == "stdin":
                    await sess.write_stdin(msg.get("data", ""))
            except json.JSONDecodeError:
                await sess.write_stdin(raw)
    except WebSocketDisconnect:
        # Don't kill the game process just because the socket closed.
      # Keep it running so the Pygame window stays alive.
        pass

@app.post("/api/shutdown")
def api_shutdown():
    stop_all_sessions()

    def _kill():
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception:
            os._exit(0)

    threading.Timer(0.2, _kill).start()
    return {"ok": True}


if __name__=="__main__":
    uvicorn.run("main:app",host="127.0.0.1",port=8000,reload=True)
