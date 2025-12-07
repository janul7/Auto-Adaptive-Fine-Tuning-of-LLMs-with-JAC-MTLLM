"""Plugin for Jac's with_llm feature for pure LLM input/output extraction + persistent fallback."""

import json
import os
import sys
import threading
import tempfile
from typing import Callable, Optional
from pathlib import Path
from datetime import datetime
import fnmatch
import re
import time, uuid
from jaclang.runtimelib.machine import hookimpl
from mtllm.llm import Model
from mtllm.mtir import MTIR


# Routing Configeration
_SWITCH_AFTER = int(os.getenv("MTLLM_SWITCH_AFTER", "1"))

# TEACHER (OpenAI)
_TEACHER_MODEL_NAME = os.getenv("MTLLM_TEACHER_MODEL", "gpt-4o-mini")
_TEACHER_API_BASE   = os.getenv("MTLLM_TEACHER_API_BASE")
_TEACHER_API_KEY    = os.getenv("MTLLM_TEACHER_API_KEY")

# STUDENT (local FastAPI model)
_STUDENT_MODEL_NAME = os.getenv("MTLLM_STUDENT_MODEL", "gpt-4o-mini")
_STUDENT_API_BASE   = os.getenv("MTLLM_STUDENT_API_BASE", "http://127.0.0.1:9000/v1")
_STUDENT_API_KEY    = os.getenv("MTLLM_STUDENT_API_KEY", "EMPTY")

# Pointer file tells the background trainer which dataset is active
_ACTIVE_LOG_POINTER = Path(os.environ.get(
    "ACTIVE_LOG_POINTER",
    str(Path.home() / ".jaseci" / "active_log.json")
))

# Route control
ROUTE_CTRL_PATH = Path(os.environ.get(
    "LLM_ROUTE_CTRL",
    str(Path.home() / ".jaseci" / "llm_route.json")
))

# Stores the last control-file signature we actually applied.
APPLIED_SIG_PATH = Path(os.environ.get(
    "LLM_ROUTE_APPLIED_SIG",
    str(Path.home() / ".jaseci" / "llm_route.applied.sig")
))

_LAST_ROUTE_CTRL_SIG: Optional[str] = None  # tracks current control-file signature

# Shared state protected with a lock.
_CALL_COUNT = 0
_LOCK = threading.Lock()


_CURRENT_MODE = "auto"

# One time model instances
_TEACHER_MODEL: Optional[Model] = None
_STUDENT_MODEL: Optional[Model] = None



# Where routing is allowed to switch to the student

_DEFAULT_SWITCH_WHITELIST = [
    r"D:\Jaseci\jac\examples\rpg_game\jac_impl\jac_impl_6\main.jac",
]

def _get_switch_whitelist() -> list[str]:
    raw = os.getenv("MTLLM_ALLOW_SWITCH_FOR", "")
    if not raw:
        return _DEFAULT_SWITCH_WHITELIST[:]
    tokens = raw.replace("|", os.pathsep).split(os.pathsep)
    return [t.strip() for t in tokens if t.strip()]

_SWITCH_WHITELIST = _get_switch_whitelist()

def _normcasepath(p: str) -> str:
    return os.path.normcase(os.path.normpath(p))

def _is_switch_allowed(run_path: Optional[str]) -> bool:
    if not run_path:
        return False
    run_norm = _normcasepath(run_path)
    for entry in _SWITCH_WHITELIST:
        e_norm = _normcasepath(entry)
        if run_norm == e_norm:
            return True
        if not e_norm.lower().endswith(".jac"):
            dir_prefix = e_norm.rstrip("\\/") + os.sep
            if run_norm.startswith(dir_prefix):
                return True
        if fnmatch.fnmatch(run_norm, e_norm):
            return True
    return False

def _resolve_run_context() -> Optional[str]:
    env_run = os.environ.get("JAC_RUN_FILE")
    if env_run:
        if not os.path.isabs(env_run):
            env_run = os.path.abspath(env_run)
        return os.path.abspath(env_run)
    try:
        for tok in reversed(sys.argv):
            if isinstance(tok, str) and tok.lower().endswith(".jac"):
                cand = tok
                if not os.path.isabs(cand):
                    cand = os.path.abspath(cand)
                if os.path.exists(cand):
                    return cand
    except Exception:
        pass
    guess = os.path.join(os.getcwd(), "main.jac")
    if os.path.exists(guess):
        return os.path.abspath(guess)
    return os.path.abspath(os.getcwd())

def _get_teacher_model() -> Model:
    global _TEACHER_MODEL
    if _TEACHER_MODEL is not None:
        return _TEACHER_MODEL
    cfg: dict[str, object] = {"verbose": True}
    if _TEACHER_API_BASE:
        cfg["api_base"] = _TEACHER_API_BASE
    if _TEACHER_API_KEY:
        cfg["api_key"] = _TEACHER_API_KEY
    _TEACHER_MODEL = Model(model_name=_TEACHER_MODEL_NAME, **cfg)
    return _TEACHER_MODEL

def _get_student_model() -> Model:
    global _STUDENT_MODEL
    if _STUDENT_MODEL is not None:
        return _STUDENT_MODEL
    _STUDENT_MODEL = Model(
        model_name=_STUDENT_MODEL_NAME,
        api_base=_STUDENT_API_BASE,
        api_key=_STUDENT_API_KEY,
        verbose=True,
    )
    return _STUDENT_MODEL

def reset_to_auto():
    global _CURRENT_MODE, _CALL_COUNT
    with _LOCK:
        _CURRENT_MODE = "auto"
        _CALL_COUNT = 0
    print("🔄 Routing mode reset to AUTO (local student will be used after switch threshold).")

def _write_active_log_pointer(dataset_jsonl_path: str, run_file_path: Optional[str] = None):
    try:
        _ACTIVE_LOG_POINTER.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset_jsonl": str(Path(dataset_jsonl_path).resolve()),
            "run_file": str(Path(run_file_path).resolve()) if run_file_path else None,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        with tempfile.NamedTemporaryFile("w", delete=False, dir=str(_ACTIVE_LOG_POINTER.parent), encoding="utf-8") as tmp:
            json.dump(payload, tmp, ensure_ascii=False)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = tmp.name
        os.replace(tmp_path, _ACTIVE_LOG_POINTER)
    except Exception as e:
        print(f"[POINTER] Failed to write active_log pointer: {e}")


# Helpers for the route-control file

def _calc_file_sig(p: Path) -> Optional[str]:
    try:
        st = p.stat()
        return f"{st.st_mtime_ns}:{st.st_size}"
    except Exception:
        return None

def _read_applied_sig() -> Optional[str]:
    try:
        if APPLIED_SIG_PATH.exists():
            return APPLIED_SIG_PATH.read_text(encoding="utf-8").strip() or None
    except Exception:
        pass
    return None

def _write_applied_sig(sig: str) -> None:
    try:
        APPLIED_SIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=str(APPLIED_SIG_PATH.parent), encoding="utf-8") as tmp:
            tmp.write(sig)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = tmp.name
        os.replace(tmp_path, APPLIED_SIG_PATH)
    except Exception as e:
        print(f"[ROUTE_CTRL] Failed writing applied signature: {e}")

def _reinit_student_model(new_api_base: Optional[str]) -> None:
    """Recreate the student singleton with a new api_base (hot switch to a different local port)."""
    global _STUDENT_API_BASE, _STUDENT_MODEL
    if not new_api_base:
        return
    if _STUDENT_API_BASE == new_api_base and _STUDENT_MODEL is not None:
        return
    old = _STUDENT_API_BASE
    _STUDENT_API_BASE = new_api_base
    _STUDENT_MODEL = Model(
        model_name=_STUDENT_MODEL_NAME,
        api_base=_STUDENT_API_BASE,
        api_key=_STUDENT_API_KEY,
        verbose=True,
    )
    print(f"[ROUTE_CTRL] Student api_base updated: {old} → {_STUDENT_API_BASE}")

def _maybe_apply_route_control() -> None:
    """
    If finetune.py has written/updated the control file, apply it atomically.
    This (a) hot-updates student api_base and (b) sets routing mode from force_mode.
    """
    global _LAST_ROUTE_CTRL_SIG, _CURRENT_MODE

    sig = _calc_file_sig(ROUTE_CTRL_PATH)
    if not sig:
        return

    # Don’t re-apply the same control on process start
    applied_sig = _read_applied_sig()
    if applied_sig == sig:
        _LAST_ROUTE_CTRL_SIG = sig
        return

    if _LAST_ROUTE_CTRL_SIG == sig:
        return

    try:
        with open(ROUTE_CTRL_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        print(f"[ROUTE_CTRL] Failed reading control file: {e}")
        return

    reason = payload.get("reason")
    force_mode = payload.get("force_mode")  # "student_forced" | "teacher_forced" | "auto"
    student_api_base = payload.get("student_api_base")

    # Only accept known reasons
    allowed_reasons = {"merge_complete", "operator_request", None}
    if reason not in allowed_reasons:
        print(f"[ROUTE_CTRL] Ignoring control (unsupported reason={reason!r}); recording signature only.")
        _LAST_ROUTE_CTRL_SIG = sig
        _write_applied_sig(sig)
        return

    # Update student target if provided
    if student_api_base:
        _reinit_student_model(student_api_base)

    # Always honor force_mode if valid
    if force_mode in ("student_forced", "teacher_forced", "auto"):
        prev = _CURRENT_MODE
        _CURRENT_MODE = "auto" if force_mode == "auto" else force_mode
        print(f"[ROUTE_CTRL] Routing mode set: {prev} → {_CURRENT_MODE} (reason={reason})")

    # Remember that we applied this version
    _LAST_ROUTE_CTRL_SIG = sig
    _write_applied_sig(sig)


# Detect the "safe default" map  (fallback case when envelope is lost)

def _is_degenerate_map(obj: object) -> bool:
    """
    Heuristic: treat as the server's safe/minimal map (fallback case)
    when arrays are empty, counts are zero, and player at (0,0).
    """
    if not isinstance(obj, dict):
        return False
    need = ("level", "walls", "small_obstacles", "enemies", "player_pos")
    if not all(k in obj for k in need):
        return False
    lvl = obj.get("level") or {}
    try:
        arrays_empty = (obj.get("walls") == [] and
                        obj.get("small_obstacles") == [] and
                        obj.get("enemies") == [])
        counts_zero = (int(lvl.get("num_wall", -1)) == 0 and
                       int(lvl.get("num_enemies", -1)) == 0)
        player_origin = (obj.get("player_pos") == {"x": 0, "y": 0})
        return arrays_empty and counts_zero and player_origin
    except Exception:
        return False


class JacMachine:
    """Jac's with_llm feature for raw LLM logging + routing + persistent fallback."""

    @staticmethod
    @hookimpl
    def call_llm(model: Model, caller: Callable, args: dict[str | int, object]) -> object:
        global _CALL_COUNT, _CURRENT_MODE

        # Pull in any new control settings before routing
        _maybe_apply_route_control()

        mtir = MTIR.factory(caller=caller, args=args, call_params=model.call_params)
        teacher = _get_teacher_model()
        student = _get_student_model()

        # Decide which model to call
        run_ctx_path = _resolve_run_context()
        switch_allowed = _is_switch_allowed(run_ctx_path)

        with _LOCK:
            _CALL_COUNT += 1
            count = _CALL_COUNT

        if _CURRENT_MODE == "teacher_forced":
            routed_model = teacher
            routed_to = "teacher(forced)"
        elif _CURRENT_MODE == "student_forced":
            routed_model = student
            routed_to = "student(forced)"
        else:
            # auto mode: teacher first, student after N calls if path is whitelisted
            if not switch_allowed:
                routed_model = teacher
                routed_to = "teacher(whitelist)"
            else:
                if count <= _SWITCH_AFTER:
                    routed_model = teacher
                    routed_to = "teacher"
                else:
                    routed_model = student
                    routed_to = "student"

        routed_model.call_params = model.call_params
        if hasattr(routed_model, "_raw_response"):
            routed_model._raw_response = None

        print(f"[ROUTER] mode={_CURRENT_MODE} route={routed_to} count={count} model={routed_model.model_name}")
        print(f"[ROUTER] api_base={routed_model.config.get('host') or routed_model.config.get('api_base')}")
        print(f"[ROUTER] run_ctx_path={run_ctx_path!r}")
        if not switch_allowed:
            print(f"[ROUTER] switching disabled: path not in whitelist")
        else:
            print(f"[ROUTER] switching allowed for this path")

        # Make the call
        try:
            result = routed_model.invoke(mtir=mtir)
        except Exception as e:
            # If the student errors out, bounce to teacher and lock mode.
            if routed_to in ("student", "student(forced)"):
                print(f"❌ Student invoke raised ({e.__class__.__name__}): switching to TEACHER and locking mode.")
                teacher.call_params = model.call_params
                if hasattr(teacher, "_raw_response"):
                    teacher._raw_response = None
                result = teacher.invoke(mtir=mtir)
                routed_model = teacher
                routed_to = "teacher(fallback-exception)"
                with _LOCK:
                    _CURRENT_MODE = "teacher_forced"
            else:
                raise  # teacher failed; surface the error

        # Fallback check
        if routed_to in ("student", "student(forced)"):
            fallback_triggered = False
            raw_text = getattr(routed_model, "_raw_response", "")

            # 1) top-level fallback
            try:
                raw_json = json.loads(raw_text) if raw_text else {}
                if isinstance(raw_json, dict):
                    if raw_json.get("fallback") is True:
                        fallback_triggered = True
                    # 1a) redundant flags
                    if not fallback_triggered:
                        chs = raw_json.get("choices")
                        if (isinstance(chs, list) and chs and isinstance(chs[0], dict)
                                and chs[0].get("mtllm_fallback") in (True, 1, "1", "true")):
                            fallback_triggered = True
                    if not fallback_triggered:
                        usage = raw_json.get("usage")
                        if isinstance(usage, dict) and usage.get("mtllm_fallback") in (True, 1, "1", "true"):
                            fallback_triggered = True
            except Exception:
                pass

            # 2) check result dict
            if not fallback_triggered and isinstance(result, dict):
                if result.get("fallback") is True:
                    fallback_triggered = True
                if not fallback_triggered:
                    chs = result.get("choices")
                    if (isinstance(chs, list) and chs and isinstance(chs[0], dict)
                            and chs[0].get("mtllm_fallback") in (True, 1, "1", "true")):
                        fallback_triggered = True
                if not fallback_triggered:
                    usage = result.get("usage")
                    if isinstance(usage, dict) and usage.get("mtllm_fallback") in (True, 1, "1", "true"):
                        fallback_triggered = True

            # 3) flattened content → detect degenerate safe map
            if not fallback_triggered:
                try:
                    obj = json.loads(raw_text) if raw_text else None
                    if isinstance(obj, dict) and _is_degenerate_map(obj):
                        fallback_triggered = True
                except Exception:
                    pass
                if not fallback_triggered and isinstance(result, dict) and _is_degenerate_map(result):
                    fallback_triggered = True

            if fallback_triggered:
                print("⚠️ Student output invalid → switching to TEACHER and locking mode.")
                teacher.call_params = model.call_params
                if hasattr(teacher, "_raw_response"):
                    teacher._raw_response = None
                result = teacher.invoke(mtir=mtir)
                routed_model = teacher
                routed_to = "teacher(fallback)"
                with _LOCK:
                    _CURRENT_MODE = "teacher_forced"

        # Build a simple JSONL log entry
        llm_msgs = mtir.get_msg_list()
        system_msg = (
            llm_msgs[0].get("content", "")
            if llm_msgs and isinstance(llm_msgs[0], dict) and llm_msgs[0].get("role") == "system"
            else ""
        )

        user_text_parts: list[str] = []
        for m in llm_msgs:
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content", "")
                if isinstance(c, str):
                    user_text_parts.append(c)
                elif isinstance(c, list):
                    for part in c:
                        if isinstance(part, dict) and part.get("type") == "text":
                            t = part.get("text", "")
                            if isinstance(t, str):
                                user_text_parts.append(t)
        user_msg = "\n".join([p for p in user_text_parts if p])

        def _normalize_json_string(s: str) -> str:
            try:
                return json.dumps(json.loads(s), ensure_ascii=False)
            except Exception:
                return s

        def _extract_assistant_content_from_obj(obj: object) -> Optional[str]:
            try:
                if isinstance(obj, str):
                    obj = json.loads(obj)
                if isinstance(obj, dict) and "choices" in obj and isinstance(obj["choices"], list) and obj["choices"]:
                    ch0 = obj["choices"][0]
                    if isinstance(ch0, dict):
                        msg = ch0.get("message") or {}
                        if isinstance(msg, dict) and "content" in msg and isinstance(msg["content"], str):
                            return msg["content"]
                        delta = ch0.get("delta") or {}
                        if isinstance(delta, dict) and "content" in delta and isinstance(delta["content"], str):
                            return delta["content"]
            except Exception:
                pass
            return None

        # Map Validation (For RPG Game only)
        _TARGET_RPG_FILE = r"D:\Jaseci\jac\examples\rpg_game\jac_impl\jac_impl_6\main.jac"

        # Try external validator first; fall back to a basic check if missing
        try:
            from .map_valid import validate_map_json as _validate_map_json
        except Exception:
            def _validate_map_json(map_obj: dict) -> bool:
                try:
                    level = map_obj["level"]
                    width = int(level["width"])
                    height = int(level["height"])
                    blocked = set()
                    for wall in map_obj.get("walls", []):
                        x1, y1 = int(wall["start_pos"]["x"]), int(wall["start_pos"]["y"])
                        x2, y2 = int(wall["end_pos"]["x"]), int(wall["end_pos"]["y"])
                        if x1 == x2:
                            for y in range(min(y1, y2), max(y1, y2)+1):
                                blocked.add((x1, y))
                        elif y1 == y2:
                            for x in range(min(x1, x2), max(x1, x2)+1):
                                blocked.add((x, y1))
                    for obs in map_obj.get("small_obstacles", []):
                        blocked.add((int(obs["x"]), int(obs["y"])))
                    from collections import deque
                    player = map_obj["player_pos"]
                    start = (int(player["x"]), int(player["y"]))
                    if not (0 <= start[0] < width and 0 <= start[1] < height):
                        return False
                    q = deque([start])
                    visited = {start}
                    dirs = [(0,1),(1,0),(0,-1),(-1,0)]
                    while q:
                        x, y = q.popleft()
                        for dx, dy in dirs:
                            nx, ny = x+dx, y+dy
                            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in blocked and (nx, ny) not in visited:
                                visited.add((nx, ny))
                                q.append((nx, ny))
                    for e in map_obj.get("enemies", []):
                        if (int(e["x"]), int(e["y"])) not in visited:
                            return False
                    return True
                except Exception:
                    return False

        def _looks_like_map(obj: object) -> bool:
            return isinstance(obj, dict) and all(k in obj for k in ("level", "walls", "small_obstacles", "enemies", "player_pos"))

        output_str: str
        raw_provider = getattr(routed_model, "_raw_response", "")
        content = _extract_assistant_content_from_obj(raw_provider)
        if content is None:
            content = _extract_assistant_content_from_obj(result)

        # Only do the retries for the specific RPG main.jac
        run_is_rpg = _normcasepath(run_ctx_path) == _normcasepath(_TARGET_RPG_FILE)
        if run_is_rpg and (content is not None or isinstance(result, dict)):
            def _try_parse_map(s: str):
                if not isinstance(s, str):
                    return None
                s = s.strip()
                try:
                    obj = json.loads(s)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    pass
                m = re.search(r"\{.*\}", s, re.S)
                if m:
                    try:
                        obj = json.loads(m.group(0))
                        return obj if isinstance(obj, dict) else None
                    except Exception:
                        pass
                return None

            parsed = _try_parse_map(content) if content is not None else None
            if parsed is None and isinstance(result, dict):
                parsed = result

            if parsed and _looks_like_map(parsed):
                MAX_RETRIES = 10
                attempts = 0
                last_content = content

                base_params = dict(model.call_params or {})
                base_temp = float(base_params.get("temperature", 0.8))
                base_temp = max(0.8, min(1.0, base_temp))
                base_top_p = float(base_params.get("top_p", 0.9)) if "top_p" in base_params else 0.9

                while attempts < MAX_RETRIES and not _validate_map_json(parsed):
                    new_params = dict(base_params)
                    new_params["temperature"] = max(0.8, min(1.0, 0.8 + 0.02 * (attempts + 1)))
                    if "top_p" in base_params:
                        new_params["top_p"] = min(0.95, base_top_p + 0.02 * (attempts + 1))
                    new_params.setdefault("presence_penalty", 0.3 + 0.1 * attempts)
                    new_params.setdefault("frequency_penalty", 0.2 + 0.1 * attempts)

                    print(f"[MAP_VALIDATOR] Map invalid → retry {attempts+1}/{MAX_RETRIES} with same input "
                          f"(temp={new_params['temperature']}"
                          f"{', top_p='+str(new_params['top_p']) if 'top_p' in new_params else ''}).")

                    mtir_retry = MTIR.factory(caller=caller, args=args, call_params=new_params)
                    routed_model.call_params = new_params
                    if hasattr(routed_model, "_raw_response"):
                        routed_model._raw_response = None

                    result = routed_model.invoke(mtir=mtir_retry)
                    raw_provider = getattr(routed_model, "_raw_response", "")
                    new_content = (
                        _extract_assistant_content_from_obj(raw_provider)
                        or _extract_assistant_content_from_obj(result)
                    )

                    if new_content == last_content:
                        print("[MAP_VALIDATOR] Retry produced identical output → bumping temperature and adding nonce, retrying again.")
                        new_params["temperature"] = max(0.8, min(1.0, new_params["temperature"] + 0.02))
                        new_params["mtllm_retry_nonce"] = f"{time.time_ns()}-{uuid.uuid4()}"
                        mtir_retry = MTIR.factory(caller=caller, args=args, call_params=new_params)
                        routed_model.call_params = new_params
                        if hasattr(routed_model, "_raw_response"):
                            routed_model._raw_response = None
                        result = routed_model.invoke(mtir=mtir_retry)
                        raw_provider = getattr(routed_model, "_raw_response", "")
                        new_content = (
                            _extract_assistant_content_from_obj(raw_provider)
                            or _extract_assistant_content_from_obj(result)
                        )

                    new_parsed = _try_parse_map(new_content) if isinstance(new_content, str) else None
                    if new_parsed is not None:
                        parsed = new_parsed

                    last_content = new_content
                    attempts += 1

                if parsed is not None:
                    try:
                        content = json.dumps(parsed, ensure_ascii=False)
                    except Exception:
                        pass

        if content is not None:
            output_str = _normalize_json_string(content)
        else:
            if isinstance(result, (dict, list)) or result is None or isinstance(result, (int, float, bool)):
                output_str = json.dumps(result, ensure_ascii=False)
            elif isinstance(result, str):
                output_str = _normalize_json_string(result)
            else:
                try:
                    output_str = json.dumps(json.loads(raw_provider), ensure_ascii=False)
                except Exception:
                    output_str = str(result)

        entry = {
            "system": system_msg,
            "input": user_msg,
            "output": output_str,
        }

        dataset_path = os.path.abspath("llm_io_log.jsonl")
        with open(dataset_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        _write_active_log_pointer(dataset_jsonl_path=dataset_path, run_file_path=run_ctx_path)

        return result

def by(model: Model) -> Callable:
    def _decorator(caller: Callable) -> Callable:
        def _wrapped(*args: object, **kwargs: object) -> object:
            invoke_args: dict[int | str, object] = {}
            for i, arg in enumerate(args):
                invoke_args[i] = arg
            for k, v in kwargs.items():
                invoke_args[k] = v
            return JacMachine.call_llm(model, caller, invoke_args)
        return _wrapped
    return _decorator
