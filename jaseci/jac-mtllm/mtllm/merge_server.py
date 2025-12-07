# merge_server.py

import os
import time
import json
import gc
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any, Union  

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_DIR = Path(os.environ.get("MODEL_DIR", r"D:\Jaseci\merged_models\jac_impl_6__123cd215\merged")).resolve()
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8010"))
DEVICE = os.environ.get("DEVICE", "cpu")  # "cpu" or "cuda"
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
DEFAULT_CONTEXT_LEN = int(os.environ.get("MAX_CONTEXT_LEN", "2048"))
LLM_ROUTE_CTRL = Path(os.environ.get("LLM_ROUTE_CTRL", str(Path.home() / ".jaseci" / "llm_route.json")))

# Pick a safe dtype for serving: fp32 on CPU, fp16 on GPU.
def _serving_dtype():
    if DEVICE == "cpu":
        return torch.float32
    return torch.float16

tokenizer = None
model = None
current_model_dir: Optional[Path] = None

def load_model(model_dir: Path):
    global tokenizer, model, current_model_dir
    if not model_dir.exists():
        raise RuntimeError(f"Model directory does not exist: {model_dir}")

    print(f"[server] Loading model from: {model_dir}")
    model_dir_posix = model_dir.as_posix()

    # Clean up any previously loaded model/tokenizer to free memory.
    if model is not None or tokenizer is not None:
        try:
            del model
            del tokenizer
        except Exception:
            pass
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        time.sleep(0.1)

    tok = AutoTokenizer.from_pretrained(model_dir_posix, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    dtype = _serving_dtype()
    print(f"[server] DEVICE={DEVICE}, dtype={dtype}")
    mdl = AutoModelForCausalLM.from_pretrained(
        model_dir_posix,
        torch_dtype=dtype,
        low_cpu_mem_usage=True if DEVICE == "cpu" else False,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    mdl.eval()

    # Swap in the new model/tokenizer.
    current_model_dir = model_dir.resolve()
    globals()["tokenizer"] = tok
    globals()["model"] = mdl
    print(f"[server] Model loaded.")

# Helper: turn various content shapes into a plain string
def _stringify_content(content: Union[str, List[Any], Dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(item if isinstance(item, str) else json.dumps(item, ensure_ascii=False))
        return "\n".join(p.strip() for p in parts if str(p).strip()).strip()
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content)

def _messages_to_prompt(messages: List[Dict[str, Any]]) -> str:
    """
    Build a prompt string from chat-style messages.

    - If the tokenizer provides a chat template, use it.
    - Otherwise, put all system messages at the top, then the rest below.
    - Accepts message.content as a string or as a list of parts.
    """
    # Normalize messages into string contents
    normalized: List[Dict[str, str]] = []
    system_accum: List[str] = []
    user_accum: List[str] = []

    for m in messages:
        role = m.get("role", "user").lower()
        content = _stringify_content(m.get("content", ""))
        normalized.append({"role": role, "content": content})

        if role == "system":
            system_accum.append(content)
        else:
            # Keep the original behavior: combine all non system parts.
            user_accum.append(content)

    system_text = "\n".join(s for s in system_accum if s.strip()).strip()
    user_text = "\n".join(u for u in user_accum if u.strip()).strip()

    # Prefer the built-in chat template when available.
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                normalized, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    # Simple fallback layout.
    if system_text:
        return f"{system_text}\n\n{user_text}\n"
    return f"{user_text}\n"

def _generate(messages: List[Dict[str, Any]],
              temperature: float = 0.8,
              top_p: float = 0.9,
              max_tokens: int = DEFAULT_MAX_NEW_TOKENS,
              stop: Optional[List[str]] = None) -> Dict[str, Any]:
    if tokenizer is None or model is None:
        raise RuntimeError("Model not loaded.")

    prompt = _messages_to_prompt(messages)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=DEFAULT_CONTEXT_LEN)
    input_len = inputs["input_ids"].shape[1]
    if DEVICE == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    do_sample = temperature is not None and temperature > 0.0
    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": do_sample,
        "temperature": float(temperature) if do_sample else None,
        "top_p": float(top_p) if do_sample else None,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    gen_ids = outputs[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    if stop:
        for s in stop:
            idx = text.find(s)
            if idx != -1:
                text = text[:idx]
                break

    return {"text": text, "prompt_tokens": input_len, "completion_tokens": len(gen_ids)}

# Try to return valid JSON when the client asks for json_schema 
def _coerce_json_to_schema(text: str, response_format: Any) -> str:
    """
    If the model already produced valid JSON, return it as-is (minified).
    If not, and response_format.type == 'json_schema', wrap the text into a
    reasonable string field that satisfies the schema.
    """
    if not response_format:
        return text

    try:
        parsed = json.loads(text)
        return json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        pass

    try:
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            schema = response_format.get("json_schema", {}).get("schema", {})
            props: Dict[str, Any] = schema.get("properties", {}) or {}
            required: List[str] = schema.get("required", []) or []

            # Prefer a required string field, otherwise any string field.
            chosen_key = None
            for k in required:
                if isinstance(props.get(k), dict) and props[k].get("type") == "string":
                    chosen_key = k
                    break
            if chosen_key is None:
                for k, v in props.items():
                    if isinstance(v, dict) and v.get("type") == "string":
                        chosen_key = k
                        break

            if chosen_key:
                return json.dumps({chosen_key: text.strip()}, separators=(",", ":"), ensure_ascii=False)

            # Last resort: put it in "output".
            return json.dumps({"output": text.strip()}, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        # If anything goes wrong, still return something valid.
        return json.dumps({"output": text.strip()}, separators=(",", ":"), ensure_ascii=False)

    return text


# FastAPI models

class ChatMessage(BaseModel):
    role: str
    # Accept string, array of parts, or dict for content.
    content: Union[str, List[Any], Dict[str, Any]]

class ChatCompletionsRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = Field(default=DEFAULT_MAX_NEW_TOKENS, alias="max_tokens")
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    response_format: Optional[Any] = None
    tools: Optional[Any] = None

class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = Field(default=DEFAULT_MAX_NEW_TOKENS, alias="max_tokens")
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


# App + background watcher for hot reloads

app = FastAPI(title="Local OpenAI-Compatible LLM", version="1.2.0")

def _file_sig(p: Path) -> Optional[str]:
    try:
        st = p.stat()
        return f"{st.st_mtime_ns}:{st.st_size}"
    except Exception:
        return None

_watch_stop = False

def _route_watcher_loop():
    """
    Check LLM_ROUTE_CTRL every 2 seconds. If it changes and includes a
    'merged_dir', load that model. Useful for live updates.
    """
    last = None
    while not _watch_stop:
        sig = _file_sig(LLM_ROUTE_CTRL)
        if sig and sig != last:
            try:
                payload = json.loads(LLM_ROUTE_CTRL.read_text(encoding="utf-8"))
                md = payload.get("merged_dir")
                if md:
                    print(f"[server] route-control changed → (re)loading: {md}")
                    load_model(Path(md))
                    print(f"[server] auto-(re)load complete.")
            except Exception as e:
                print(f"[server] auto-reload failed: {e}")
            last = sig
        time.sleep(2.0)

@app.on_event("startup")
def _startup():
    # Start the watcher thread first.
    threading.Thread(target=_route_watcher_loop, daemon=True).start()

    # Try to load from MODEL_DIR. If that fails, try the route-control file.
    # If neither works, keep running and wait for a valid merge to appear.
    target = MODEL_DIR
    if target.exists():
        try:
            load_model(target)
            return
        except Exception as e:
            print(f"[server] initial load from MODEL_DIR failed: {e}")

    if LLM_ROUTE_CTRL.exists():
        try:
            payload = json.loads(LLM_ROUTE_CTRL.read_text(encoding="utf-8"))
            md = payload.get("merged_dir")
            if md and Path(md).exists():
                load_model(Path(md))
                return
            else:
                print("[server] LLM_ROUTE_CTRL found but merged_dir missing or does not exist. Waiting...")
        except Exception as e:
            print(f"[server] could not parse LLM_ROUTE_CTRL: {e}")

    # Nothing to load yet, the server will stay up and wait.
    print("[server] No model to load yet. Waiting for first successful merge...")

@app.on_event("shutdown")
def _shutdown():
    global _watch_stop
    _watch_stop = True

@app.get("/v1/health")
def health():
    status = "ready" if (model is not None and tokenizer is not None) else "loading"
    return {"status": status, "model_dir": str(current_model_dir) if current_model_dir else None}

@app.get("/v1/models")
def list_models():
    if model is None:
        return {"object": "list", "data": []}
    name = current_model_dir.name if current_model_dir else "merged"
    return {"object": "list", "data": [{"id": name, "object": "model"}]}

@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionsRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Waiting for first merge.")
    if req.stream:
        raise HTTPException(status_code=400, detail="stream=true not supported.")
    try:
        result = _generate(
            messages=[m.dict() for m in req.messages],
            temperature=req.temperature or 0.0,
            top_p=req.top_p or 1.0,
            max_tokens=req.max_tokens or DEFAULT_MAX_NEW_TOKENS,
            stop=req.stop,
        )

        out_text = result["text"]
        # If a JSON response was requested, try to return valid JSON.
        out_content = _coerce_json_to_schema(out_text, req.response_format)

        return {
            "id": f"chatcmpl-local-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or (current_model_dir.name if current_model_dir else "merged"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": out_content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions")
def completions(req: CompletionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Waiting for first merge.")
    messages = [{"role": "user", "content": req.prompt}]
    return chat_completions(ChatCompletionsRequest(
        model=req.model,
        messages=[ChatMessage(**m) for m in messages],
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
        stream=req.stream,
        stop=req.stop,
    ))

@app.post("/v1/reload")
def reload_model(payload: Dict[str, Any]):
    target = payload.get("merged_dir")
    if not target and LLM_ROUTE_CTRL.exists():
        try:
            data = json.loads(LLM_ROUTE_CTRL.read_text(encoding="utf-8"))
            target = data.get("merged_dir")
        except Exception:
            pass
    if not target:
        raise HTTPException(status_code=400, detail="No merged_dir provided and route control not found.")
    try:
        load_model(Path(target))
        return {"status": "ok", "merged_dir": str(current_model_dir)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=HOST, port=PORT, reload=False)
