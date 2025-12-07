# Server.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union, List, Any, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import json

# Import the validator that checks whether the model's output is acceptable
from Pattern_Validator import validate_text

MODEL_PATH = r"D:\Jac\TinyLlama-1.1B-Chat-v1.0-merged_7"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto"
)
print("✅ Model loaded successfully.")

app = FastAPI()

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Any], Dict[str, Any]]

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = 2000
    temperature: float = 0.7
    response_format: Optional[Any] = None
    tools: Optional[Any] = None

# Helpers
def stringify_content(content: Union[str, List[Any], Dict[str, Any]]) -> str:
    """
    Turn incoming message content (string, list, or dict) into a clean string.
    - Lists are joined line by line.
    - Dicts are JSON-encoded.
    """
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

def strip_code_fences(s: str) -> str:
    """
    Remove leading/trailing Markdown code fences if present.
    """
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
    s = s.strip()
    if s.endswith("```"):
        s = s.rsplit("\n", 1)[0]
    return s.strip()

def coerce_to_single_json_object_text(raw: str) -> str:
    """
    Best-effort: try to return a compact JSON object string.
    - If the text parses as JSON and contains an 'output' field with a JSON string,
      unwrap it.
    - If a 'time' field is a float, round it down to an int.
    - If parsing fails, return the original text.
    """
    def fix_time_key(obj: dict) -> dict:
        if "time" in obj and isinstance(obj["time"], float):
            obj["time"] = int(obj["time"])
        return obj
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            if "output" in parsed:
                inner = parsed["output"]
                if isinstance(inner, str):
                    try:
                        inner_obj = json.loads(inner)
                        if isinstance(inner_obj, dict):
                            inner_obj = fix_time_key(inner_obj)
                            return json.dumps(inner_obj, separators=(",", ":"), ensure_ascii=False)
                    except Exception:
                        pass
                if isinstance(inner, dict):
                    inner = fix_time_key(inner)
                    return json.dumps(inner, separators=(",", ":"), ensure_ascii=False)
            parsed = fix_time_key(parsed)
            return json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)
        return raw
    except Exception:
        return raw

def _safe_minimal_map_str() -> str:
    """
    Return a minimal, safe default map payload as a compact JSON string.
    Used when validation fails.
    """
    stub = {
        "level": {
            "name": 0,
            "difficulty": 1,
            "time": 60,
            "width": 8,
            "height": 8,
            "num_wall": 0,
            "num_enemies": 0
        },
        "walls": [],
        "small_obstacles": [],
        "enemies": [],
        "player_pos": {"x": 0, "y": 0}
    }
    return json.dumps(stub, separators=(",", ":"), ensure_ascii=False)

def _build_response(
    req: ChatRequest,
    inputs,
    outputs,
    content_str: str,
    fallback_flag: bool
) -> dict:
    """
    Build an OpenAI-style response dict.
    We mirror the 'fallback' flag in multiple places so different wrappers
    (that read different parts of the response) can still detect it.
    """
    resp = {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            # also expose the fallback flag at the choice level
            "mtllm_fallback": bool(fallback_flag),
            "message": {"role": "assistant", "content": content_str},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "completion_tokens": int(len(outputs[0]) - inputs["input_ids"].shape[1]),
            "total_tokens": int(len(outputs[0])),
            # duplicate the flag here for clients that only read usage
            "mtllm_fallback": 1 if fallback_flag else 0,
        },
        # main place to check for fallback
        "fallback": bool(fallback_flag),
    }
    return resp

# Route 
@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    # Gather system and user messages into plain strings
    system_accum: List[str] = []
    user_accum: List[str] = []
    for msg in req.messages:
        text = stringify_content(msg.content)
        role = msg.role.lower()
        if role == "system":
            system_accum.append(text)
        elif role == "user":
            user_accum.append(text)
        else:
            user_accum.append(text)

    system_text = "\n".join(s for s in system_accum if s.strip()).strip()
    user_text = "\n".join(u for u in user_accum if u.strip()).strip()

    # Print the prompt we send to the local model (for debugging/visibility)
    prompt_obj = {"system": system_text, "input": user_text}
    print("\n================ PROMPT SENT TO MODEL ================\n")
    print(json.dumps(prompt_obj, ensure_ascii=False))
    print("\n=======================================================\n")

    # Build the raw text prompt for the model
    prompt_for_model = f"{system_text}\n\n{user_text}\n" if system_text else f"{user_text}\n"
    inputs = tokenizer(prompt_for_model, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=True,
            temperature=float(req.temperature),
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    # Clean up the model text and run validation
    normalized_text = strip_code_fences(generated_text)
    validation = validate_text(normalized_text)

    if validation.get("kind") == "INVALID":
        # If the output doesn't pass, return a safe fallback map
        # and mark the response as a fallback in multiple places.
        print("❌ Validation failed. Returning safe MAP + fallback flags.")
        safe_map_content = _safe_minimal_map_str()
        return _build_response(
            req=req,
            inputs=inputs,
            outputs=outputs,
            content_str=safe_map_content,
            fallback_flag=True
        )

    # If valid, try to return a single compact JSON object string
    content_str = coerce_to_single_json_object_text(normalized_text)
    return _build_response(
        req=req,
        inputs=inputs,
        outputs=outputs,
        content_str=content_str,
        fallback_flag=False
    )
