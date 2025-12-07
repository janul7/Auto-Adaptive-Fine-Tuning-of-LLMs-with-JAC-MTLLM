# Incremental_FineTuning.py

# Our script combines two approaches:
#   - RPG trainer (with validation checks and milestones)
#   - Generic trainer (for normal .jac datasets in chunks)


# Turn off mixed-precision before loading HF libs
import os
os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
for _k in ("HF_FP16", "HF_BF16", "FP16", "BF16", "MIXED_PRECISION"):
    os.environ.pop(_k, None)

import time
import json
import shutil
import tempfile
import hashlib
import random
import gc
import stat
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from json import JSONDecodeError

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.nn import MSELoss

# Try to load the RPG-specific validator
try:
    from Pattern_Validator import validate_text
except Exception as _e:
    validate_text = None
    print(f"[WARN] pattern_validator not importable: {_e}")

# Custom TrainingArguments to make sure FP32 is used
class FP32TrainingArguments(TrainingArguments):
    def __post_init__(self):
        self.fp16 = False
        self.bf16 = False
        super().__post_init__()


# User Settings

INITIAL_BASE_MODEL = os.environ.get("INITIAL_BASE_MODEL", r"D:\Jac\TinyLlama-1.1B-Chat-v1.0")

WORK_ROOT   = Path(os.environ.get("WORK_ROOT",   r"D:\Jaseci\finetune_runs")).resolve()
MERGED_ROOT = Path(os.environ.get("MERGED_ROOT", str(WORK_ROOT.parent / "merged_models"))).resolve()

GEN_CHUNK_SIZE   = int(os.environ.get("CHUNK_SIZE", "5"))
MAX_SEQ_LEN      = int(os.environ.get("MAX_SEQ_LEN", "512"))
NUM_EPOCHS       = int(os.environ.get("NUM_EPOCHS", "1"))
BATCH_SIZE       = int(os.environ.get("BATCH_SIZE", "1"))
LEARNING_RATE    = float(os.environ.get("LEARNING_RATE", "1e-3"))
SLEEP_SECONDS_WHEN_IDLE = int(os.environ.get("SLEEP_SECONDS_WHEN_IDLE", "20"))

MILESTONE_ROWS          = int(os.environ.get("MILESTONE_ROWS", "10"))
VALIDATION_SAMPLE_ROWS  = int(os.environ.get("VALIDATION_SAMPLE_ROWS", "4"))
RANDOM_SEED             = int(os.environ.get("RANDOM_SEED", "1337"))
random.seed(RANDOM_SEED)

MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "2000"))
TEMPERATURE    = float(os.environ.get("TEMPERATURE", "0.0"))
TOP_P          = float(os.environ.get("TOP_P", "1.0"))
DO_SAMPLE      = bool(int(os.environ.get("DO_SAMPLE", "0")))

def is_low_usage() -> bool:
    return True


# Pointer Tracking

POINTER_PATH = Path(os.environ.get("ACTIVE_LOG_POINTER", str(Path.home() / ".jaseci" / "active_log.json")))

WORK_ROOT.mkdir(parents=True, exist_ok=True)
MERGED_ROOT.mkdir(parents=True, exist_ok=True)
STATE_PATH = WORK_ROOT / "state.json"

# Helper to read the dataset currently being worked on
def read_active_dataset_from_pointer() -> Optional[Dict[str, Any]]:
    try:
        if not POINTER_PATH.exists():
            return None
        with open(POINTER_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        p = data.get("dataset_jsonl")
        if not p:
            return None
        pth = Path(p)
        return data if pth.exists() else None
    except (OSError, JSONDecodeError):
        return None

# Load progress state (datasets processed, etc.)
def load_state() -> Dict[str, Any]:
    st: Dict[str, Any] = {"initialized": False, "datasets": {}}
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                st.update(loaded)
        except Exception:
            pass
    st.setdefault("initialized", False)
    st.setdefault("datasets", {})
    return st

# Save progress
def save_state(state: Dict[str, Any]) -> None:
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

# Helpers for tracking which rows have been processed
def get_last_processed(state: Dict[str, Any], dataset_path: str) -> int:
    return int(state["datasets"].get(dataset_path, {}).get("last_processed", 0))

def set_last_processed(state: Dict[str, Any], dataset_path: str, idx: int) -> None:
    state["datasets"].setdefault(dataset_path, {})
    state["datasets"][dataset_path]["last_processed"] = int(idx)
    save_state(state)


# RPG Detection

_RPG_RUN_WHITELIST = [r"D:\Jaseci\jac\examples\rpg_game\jac_impl\jac_impl_6\main.jac"]
_RPG_HINTS = [h.strip().lower() for h in os.getenv("RPG_RUN_HINTS", "rpg_game").split(os.pathsep) if h.strip()]

# Normalize path for safe comparison
def _normcasepath(p: str) -> str:
    return os.path.normcase(os.path.normpath(p or ""))

# Detect if this looks like an RPG run
def _looks_like_rpg_by_whitelist(run_file: str, dataset_path: str) -> bool:
    rf = _normcasepath(run_file)
    for w in _RPG_RUN_WHITELIST:
        wn = _normcasepath(w)
        if rf == wn:
            return True
        if rf.startswith(os.path.dirname(wn)):
            return True
    dp = _normcasepath(dataset_path)
    return "rpg_game" in dp

def _looks_like_rpg_by_hints(run_file: str, dataset_path: str) -> bool:
    run = (run_file or "").lower().replace("\\", "/")
    ds  = (dataset_path or "").lower().replace("\\", "/")
    return any(h in run or h in ds for h in _RPG_HINTS)

def looks_like_rpg(run_file: str, dataset_path: str) -> bool:
    return _looks_like_rpg_by_whitelist(run_file, dataset_path) or _looks_like_rpg_by_hints(run_file, dataset_path)


# Shared Helpers 

def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]

def get_dataset_len(dataset_jsonl_path: str) -> int:
    ds = load_dataset("json", data_files=dataset_jsonl_path, split="train")
    return len(ds)

def _pf(p: Path) -> str:
    return p.resolve().as_posix()

# Debugging
def _debug_list_dir(p: Path, label: str):
    try:
        items = sorted(x.name for x in p.iterdir())
        print(f"[{datetime.now()}] {label} exists={p.exists()} files={items[:12]}")
    except Exception as e:
        print(f"[{datetime.now()}] {label} list failed: {e}")

# Combine fields into training text
def build_text(sample: Dict[str, Any]) -> str:
    system = (sample.get("system") or "").strip()
    user_in = (sample.get("input") or "").strip()
    out     = (sample.get("output") or "").strip()
    return (system + "\n" + user_in + "\n" + out).strip()

# Prompt-masked preprocessing (mask system+input; learn on output only)
def preprocess_dataset(ds, tokenizer):
    def _map_fn(sample):
        system = (sample.get("system") or "").strip()
        user_in = (sample.get("input") or "").strip()
        out     = (sample.get("output") or "").strip()

        prompt = (system + "\n" + user_in + "\n").strip()
        if prompt:
            prompt = prompt + "\n"
        full_text = (prompt + out).strip()

        tok_full = tokenizer(full_text, max_length=MAX_SEQ_LEN, truncation=True, padding="max_length")
        tok_prompt = tokenizer(prompt, max_length=MAX_SEQ_LEN, truncation=True, padding=False)
        labels = tok_full["input_ids"].copy()

        prompt_len = min(len(tok_prompt["input_ids"]), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        tok_full["labels"] = labels
        return tok_full

    cols = ds.column_names
    return ds.map(_map_fn, remove_columns=cols)

# Trainer with extra regularization (keeps model close to ref)
class RegularizedTrainer(Trainer):
    def __init__(self, *args, ref_model=None, reg_lambda=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.reg_lambda = reg_lambda
        self.mse_loss = MSELoss()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        if self.ref_model is not None:
            ref_params = dict(self.ref_model.named_parameters())
            reg_loss = 0.0
            for n, p in model.named_parameters():
                if n in ref_params and p.requires_grad:
                    reg_loss = reg_loss + self.mse_loss(p, ref_params[n].detach())
            loss = loss + self.reg_lambda * reg_loss
        return (loss, outputs) if return_outputs else loss

# File cleanup helpers (Windows-safe)
def _on_rm_error(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass
    try:
        func(path)
    except Exception:
        pass

def _rmtree_retry(path: Path, retries: int = 8, delay: float = 0.4) -> None:
    for _ in range(retries):
        try:
            shutil.rmtree(path, onerror=_on_rm_error)
            return
        except Exception:
            time.sleep(delay)
    shutil.rmtree(path, ignore_errors=True)

def _atomic_replace_dir(src: Path, dst: Path, retries: int = 8, delay: float = 0.4) -> None:
    if dst.exists():
        _rmtree_retry(dst, retries=retries, delay=delay)
        time.sleep(0.15)
    for _ in range(retries):
        try:
            os.replace(str(src), str(dst))
            return
        except (PermissionError, OSError):
            time.sleep(delay)
    if not dst.exists():
        shutil.copytree(str(src), str(dst))
    shutil.rmtree(str(src), ignore_errors=True)


# Route Control

ROUTE_CTRL_PATH = Path(os.environ.get("LLM_ROUTE_CTRL", str(Path.home() / ".jaseci" / "llm_route.json")))

# Write routing info after merge (so runtime knows which model to use)
def write_route_control(force_mode: str = "student_forced",
                        student_api_base: Optional[str] = None,
                        merged_dir: Optional[Path] = None,
                        project_tag: Optional[str] = None,
                        reason: str = "merge_complete"):
    try:
        ROUTE_CTRL_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "force_mode": force_mode,
            "student_api_base": student_api_base,
            "merged_dir": str(merged_dir) if merged_dir else None,
            "project_tag": project_tag,
            "reason": reason,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        with tempfile.NamedTemporaryFile("w", delete=False, dir=str(ROUTE_CTRL_PATH.parent), encoding="utf-8") as tmp:
            json.dump(payload, tmp, ensure_ascii=False)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = tmp.name
        os.replace(tmp_path, ROUTE_CTRL_PATH)
        print(f"[ROUTE_CTRL] Wrote control file → {ROUTE_CTRL_PATH}")
    except Exception as e:
        print(f"[ROUTE_CTRL] Failed to write control file: {e}")


# RPG PIPELINE (milestones + gate)

def rpg_project_paths_for_dataset(dataset_jsonl: Path) -> Tuple[str, Path, Path]:
    project_dir = dataset_jsonl.resolve().parent
    tag = f"{project_dir.name}__{_short_hash(str(project_dir))}"
    project_work_root = WORK_ROOT / tag
    project_merged_dir = MERGED_ROOT / tag / "merged"
    project_work_root.mkdir(parents=True, exist_ok=True)
    project_merged_dir.parent.mkdir(parents=True, exist_ok=True)
    return tag, project_work_root, project_merged_dir

# quick check if a folder looks like a model directory
def _is_valid_model_dir(p: Path) -> bool:
    return p.exists() and ((p / "config.json").exists()) and ((p / "pytorch_model.bin").exists() or (p / "model.safetensors").exists())

# make sure merged model folder is ready (copy base if missing)
def rpg_ensure_merged_exists(merged_dir: Path):
    if _is_valid_model_dir(merged_dir):
        return
    print(f"[{datetime.now()}] Initializing merged model at: {merged_dir}")
    if merged_dir.exists():
        _rmtree_retry(merged_dir)
        time.sleep(0.2)
    src = Path(INITIAL_BASE_MODEL)
    if not _is_valid_model_dir(src):
        raise RuntimeError(f"INITIAL_BASE_MODEL invalid: {src}")
    merged_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, merged_dir)

# load only the first n rows of dataset
def load_dataset_first_n(dataset_jsonl_path: str, n: int):
    ds = load_dataset("json", data_files=dataset_jsonl_path, split="train")
    n = min(n, len(ds))
    indices = list(range(n))
    return ds.select(indices)

# turn a dataset row into a prompt
def build_prompt_from_row(row: Dict[str, Any]) -> str:
    system = (row.get("system") or "").strip()
    user_in = (row.get("input") or "").strip()
    return f"{system}\n{user_in}\n"

# train LoRA adapter on dataset (full milestone)
def rpg_train_lora_on_dataset(base_model_dir: Path, ds, run_dir: Path) -> Path:
    try:
        base_model_dir.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    rpg_ensure_merged_exists(base_model_dir)

    _debug_list_dir(base_model_dir, "Base/Merged model dir before load")

    print(f"[{datetime.now()}] Loading tokenizer/model from: {base_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(_pf(base_model_dir), use_fast=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(_pf(base_model_dir), device_map="cpu", torch_dtype=torch.float32, low_cpu_mem_usage=False, local_files_only=True)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    ref_model = AutoModelForCausalLM.from_pretrained(_pf(base_model_dir), device_map="cpu", torch_dtype=torch.float32, local_files_only=True)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj"], bias="none")
    model = get_peft_model(model, lora_config)

    tokenized = preprocess_dataset(ds, tokenizer)

    print("[precision] Forcing FP32: fp16=False, bf16=False")
    training_args = FP32TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_dir=str(run_dir / "logs"),
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
    )

    trainer = RegularizedTrainer(model=model, args=training_args, train_dataset=tokenized, ref_model=ref_model, reg_lambda=0.01)
    print(f"[{datetime.now()}] Starting LoRA training on dataset (rows={len(ds)}) ...")
    trainer.train()

    delta_dir = run_dir / "delta"
    delta_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{datetime.now()}] Saving LoRA delta to: {delta_dir}")
    trainer.model.save_pretrained(str(delta_dir))
    tokenizer.save_pretrained(str(delta_dir))

    latest = run_dir.parent / "adapter_latest"
    if latest.exists():
        _rmtree_retry(latest)
    shutil.copytree(str(delta_dir), str(latest))
    return delta_dir

# merge LoRA delta into main merged model
def rpg_merge_delta_into_merged(delta_dir: Path, merged_dir: Path):
    print(f"[{datetime.now()}] Merging delta into merged model at: {merged_dir}")
    tok = AutoTokenizer.from_pretrained(_pf(merged_dir), use_fast=True, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(_pf(merged_dir), device_map="cpu", torch_dtype=torch.float16, local_files_only=True)

    model = PeftModel.from_pretrained(base, str(delta_dir))
    model = model.merge_and_unload().to(torch.float16)

    parent = merged_dir.parent
    tmp_out = Path(tempfile.mkdtemp(prefix="merge_tmp_", dir=str(parent)))
    model.save_pretrained(str(tmp_out))
    tok.save_pretrained(str(tmp_out))

    del model, base, tok
    gc.collect()
    time.sleep(0.2)

    _atomic_replace_dir(tmp_out, merged_dir)
    print(f"[{datetime.now()}] Merge complete. Updated merged model at: {merged_dir}")

# read first n lines from a jsonl file
def _read_jsonl_first_n(dataset_path: Path, n: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n: break
            s = line.strip()
            if not s: continue
            try:
                obj = json.loads(s)
            except Exception:
                obj = {}
            out.append(obj)
    return out

# load base + adapter for inference (not merged)
def _load_model_with_delta_for_infer(base_model_dir: Path, delta_dir: Path):
    print(f"[GATE][INFER] Loading merged base for inference: {base_model_dir}")
    print(f"[GATE][INFER] Attaching fresh adapter (not merged): {delta_dir}")

    tokenizer = AutoTokenizer.from_pretrained(_pf(base_model_dir), use_fast=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    base = AutoModelForCausalLM.from_pretrained(_pf(base_model_dir), device_map=None, torch_dtype=dtype, low_cpu_mem_usage=False, local_files_only=True).to(device)
    model = PeftModel.from_pretrained(base, str(delta_dir))
    model.eval()
    return tokenizer, model, device

# run one prompt through model
def _generate_single(model, tokenizer, device, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = gen[0][input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()

# run gate validation by sampling rows and checking outputs
def validator_gate_inference(dataset_path: Path, milestone_rows: int, sample_k: int,
                             base_model_dir: Path, delta_dir: Path) -> bool:
    print("[GATE][INFER] Starting inference-driven validation...")
    if validate_text is None:
        print("[GATE][INFER] Validator not available; FAILING gate for safety.")
        return False

    rows = _read_jsonl_first_n(dataset_path, milestone_rows)
    total = len(rows)
    if total == 0:
        print("[GATE][INFER] No rows to validate.")
        return False

    k = min(sample_k, total)
    indices = random.sample(range(total), k)
    print(f"[GATE][INFER] Sampling {k} rows out of first {milestone_rows} (available={total}) -> indices={indices}")

    try:
        tokenizer, model, device = _load_model_with_delta_for_infer(base_model_dir, delta_dir)
    except Exception as e:
        print(f"[GATE][INFER] Failed to load model+delta for inference: {e}")
        return False

    all_ok = True
    for idx in indices:
        row = rows[idx]
        prompt = build_prompt_from_row(row)
        try:
            gen_text = _generate_single(model, tokenizer, device, prompt)
        except Exception as e:
            print(f"[GATE][INFER] Row {idx+1}: generation failed -> {e}")
            all_ok = False
            continue

        result = validate_text(gen_text)
        kind = result.get("kind")
        strict = result.get("strict", False)
        if kind in ("LEVEL", "MAP"):
            print(f"[GATE][INFER] Row {idx+1}: kind={kind}, strict={strict} -> OK")
        else:
            print(f"[GATE][INFER] Row {idx+1}: INVALID -> reason={result.get('reason')}")
            all_ok = False

    print(f"[GATE][INFER] Result: {'PASS' if all_ok else 'FAIL'}")
    return all_ok

# figure out the next milestone point
def compute_next_milestone(total_rows: int, last_processed: int, unit: int) -> Optional[int]:
    if total_rows < unit:
        return None
    candidate = (total_rows // unit) * unit
    if candidate > last_processed:
        return candidate
    return None


# GENERIC PIPELINE (chunk trainer)

def gen_project_paths_for_dataset(dataset_jsonl: Path) -> Tuple[str, Path, Path]:
    project_dir = dataset_jsonl.resolve().parent
    tag = f"{project_dir.name}__{_short_hash(str(project_dir))}"
    project_work_root = WORK_ROOT / tag
    project_merged_dir = MERGED_ROOT / tag / "merged"
    project_work_root.mkdir(parents=True, exist_ok=True)
    project_merged_dir.parent.mkdir(parents=True, exist_ok=True)
    return tag, project_work_root, project_merged_dir

# make sure generic merged folder is ready
def gen_ensure_merged_exists(merged_dir: Path):
    if merged_dir.exists() and any(merged_dir.iterdir()):
        return
    print(f"[{datetime.now()}] Initializing merged model at: {merged_dir}")
    if merged_dir.exists():
        _rmtree_retry(merged_dir)
        time.sleep(0.2)
    shutil.copytree(INITIAL_BASE_MODEL, merged_dir)

# load dataset rows for chunk training
def load_chunk_dataset(dataset_jsonl_path: str, start_idx: int, end_idx: int):
    ds = load_dataset("json", data_files=dataset_jsonl_path, split="train")
    indices = list(range(start_idx - 1, end_idx))
    return ds.select(indices)

# train LoRA adapter on a chunk of dataset
def gen_train_lora_on_chunk(base_model_dir: Path, chunk_ds, run_dir: Path) -> Path:
    print(f"[{datetime.now()}] Loading tokenizer/model from: {base_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(str(base_model_dir), device_map="cpu", torch_dtype=torch.float32, low_cpu_mem_usage=False)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    ref_model = AutoModelForCausalLM.from_pretrained(str(base_model_dir), device_map="cpu", torch_dtype=torch.float32)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj"], bias="none")
    model = get_peft_model(model, lora_config)

    tokenized = preprocess_dataset(chunk_ds, tokenizer)

    print("[precision] Forcing FP32: fp16=False, bf16=False")
    training_args = FP32TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_dir=str(run_dir / "logs"),
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
    )

    trainer = RegularizedTrainer(model=model, args=training_args, train_dataset=tokenized, ref_model=ref_model, reg_lambda=0.01)
    print(f"[{datetime.now()}] Starting LoRA training on chunk...")
    trainer.train()

    delta_dir = run_dir / "delta"
    delta_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{datetime.now()}] Saving LoRA delta to: {delta_dir}")
    trainer.model.save_pretrained(str(delta_dir))
    tokenizer.save_pretrained(str(delta_dir))

    latest = run_dir.parent / "adapter_latest"
    if latest.exists():
        _rmtree_retry(latest)
    shutil.copytree(str(delta_dir), str(latest))
    return delta_dir

# merge LoRA delta into generic merged model
def gen_merge_delta_into_merged(delta_dir: Path, merged_dir: Path):
    print(f"[{datetime.now()}] Merging delta into merged model at: {merged_dir}")

    tok = AutoTokenizer.from_pretrained(str(merged_dir), use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        str(merged_dir),
        device_map="cpu",
        torch_dtype=torch.float16,
    )

    model = PeftModel.from_pretrained(base, str(delta_dir))
    model = model.merge_and_unload().to(torch.float16)

    parent = merged_dir.parent
    tmp_out = Path(tempfile.mkdtemp(prefix="merge_tmp_", dir=str(parent)))
    model.save_pretrained(str(tmp_out))
    tok.save_pretrained(str(tmp_out))

    del model, base, tok
    gc.collect()
    time.sleep(0.2)

    _atomic_replace_dir(tmp_out, merged_dir)
    print(f"[{datetime.now()}] Merge complete. Updated merged model at: {merged_dir}")

 
# Main Loop
 
def main():
    state = load_state()

    # RPG trackers
    rpg_current_dataset_path: Optional[str] = None
    rpg_current_project_tag: Optional[str] = None
    rpg_current_project_work_root: Optional[Path] = None
    rpg_current_project_merged_dir: Optional[Path] = None
    rpg_run_counter: Dict[str, int] = {}

    # Generic trackers
    gen_current_dataset_path: Optional[str] = None
    gen_current_project_tag: Optional[str] = None
    gen_current_project_work_root: Optional[Path] = None
    gen_current_project_merged_dir: Optional[Path] = None
    gen_run_counter: Dict[str, int] = {}

    while True:
        try:
            pointer = read_active_dataset_from_pointer()
            if not pointer:
                print(f"[{datetime.now()}] Waiting for pointer at: {POINTER_PATH}")
                time.sleep(SLEEP_SECONDS_WHEN_IDLE)
                continue

            dataset_path = str(Path(pointer["dataset_jsonl"]).resolve())
            run_file = pointer.get("run_file") or ""

            if looks_like_rpg(run_file, dataset_path):
                # RPG PIPELINE
                if rpg_current_dataset_path != dataset_path:
                    tag, proj_work, proj_merged = rpg_project_paths_for_dataset(Path(dataset_path))
                    rpg_current_dataset_path = dataset_path
                    rpg_current_project_tag = tag
                    rpg_current_project_work_root = proj_work
                    rpg_current_project_merged_dir = proj_merged
                    rpg_run_counter.setdefault(tag, 0)
                    print(f"[{datetime.now()}] Switching active dataset to: {dataset_path}")
                    print(f"[{datetime.now()}] Project tag: {tag}")

                    # Ensure initial merged exists immediately
                    rpg_ensure_merged_exists(rpg_current_project_merged_dir)

                total_rows = get_dataset_len(rpg_current_dataset_path)
                last_processed = get_last_processed(state, rpg_current_dataset_path)

                if total_rows < last_processed:
                    print(f"[{datetime.now()}] Detected dataset reset/rotation. total={total_rows} < last={last_processed}. Resetting progress.")
                    last_processed = 0
                    set_last_processed(state, rpg_current_dataset_path, last_processed)

                milestone = compute_next_milestone(total_rows, last_processed, MILESTONE_ROWS)
                if milestone is None or not is_low_usage():
                    print(f"[{datetime.now()}] Waiting... dataset={rpg_current_dataset_path}, total={total_rows}, last_processed={last_processed}, unit={MILESTONE_ROWS}")
                    time.sleep(SLEEP_SECONDS_WHEN_IDLE)
                    continue

                print(f"[{datetime.now()}] Preparing milestone {milestone} (train on first {milestone} rows) / total={total_rows} for dataset={rpg_current_dataset_path}")

                # Re ensure merged right before training
                rpg_current_project_merged_dir.parent.mkdir(parents=True, exist_ok=True)
                rpg_ensure_merged_exists(rpg_current_project_merged_dir)

                try:
                    ds = load_dataset_first_n(rpg_current_dataset_path, milestone)
                except Exception as e:
                    print(f"[ERROR] Failed to load first {milestone} rows from {rpg_current_dataset_path}: {e}")
                    time.sleep(SLEEP_SECONDS_WHEN_IDLE)
                    continue

                rpg_run_counter[rpg_current_project_tag] += 1
                run_dir = rpg_current_project_work_root / f"run_{rpg_run_counter[rpg_current_project_tag]:03d}"
                run_dir.mkdir(parents=True, exist_ok=True)

                try:
                    delta_dir = rpg_train_lora_on_dataset(rpg_current_project_merged_dir, ds, run_dir)
                except Exception as e:
                    print(f"[ERROR] Training failed for {rpg_current_dataset_path}: {e}")
                    time.sleep(SLEEP_SECONDS_WHEN_IDLE)
                    continue

                try:
                    gate_ok = validator_gate_inference(Path(rpg_current_dataset_path), milestone, VALIDATION_SAMPLE_ROWS, rpg_current_project_merged_dir, delta_dir)
                except Exception as e:
                    print(f"[ERROR] Validator gate crashed: {e}")
                    gate_ok = False

                if gate_ok:
                    try:
                        rpg_merge_delta_into_merged(delta_dir, rpg_current_project_merged_dir)
                        print(f"[{datetime.now()}] Milestone {milestone}: merge SUCCESS.")
                        # Route control after successful merge
                        write_route_control(
                            force_mode=os.environ.get("CTRL_FORCE_MODE", "student_forced"),
                            student_api_base=os.environ.get("MTLLM_STUDENT_API_BASE", "http://127.0.0.1:8010/v1"),
                            merged_dir=rpg_current_project_merged_dir,
                            project_tag=rpg_current_project_tag,
                            reason="merge_complete"
                        )
                    except Exception as e:
                        print(f"[ERROR] Merge failed for milestone {milestone}: {e}")
                        time.sleep(SLEEP_SECONDS_WHEN_IDLE)
                        continue
                else:
                    print(f"[{datetime.now()}] Milestone {milestone}: validator gate FAILED. Skipping merge; adapters kept on disk only.")

                set_last_processed(state, rpg_current_dataset_path, milestone)
                print(f"[{datetime.now()}] Completed milestone {milestone} for dataset={rpg_current_dataset_path}. Current merged: {rpg_current_project_merged_dir}")

            else:
                # GENERIC PIPELINE
                if gen_current_dataset_path != dataset_path:
                    tag, proj_work, proj_merged = gen_project_paths_for_dataset(Path(dataset_path))
                    gen_current_dataset_path = dataset_path
                    gen_current_project_tag = tag
                    gen_current_project_work_root = proj_work
                    gen_current_project_merged_dir = proj_merged
                    gen_run_counter.setdefault(tag, 0)
                    print(f"[{datetime.now()}] Switching active dataset to: {dataset_path}")
                    print(f"[{datetime.now()}] Project tag: {tag}")

                    # Ensure initial merged exists immediately
                    gen_ensure_merged_exists(gen_current_project_merged_dir)

                total_rows = get_dataset_len(gen_current_dataset_path)
                last_processed = get_last_processed(state, gen_current_dataset_path)

                if total_rows < last_processed:
                    print(f"[{datetime.now()}] Detected dataset reset/rotation. total={total_rows} < last={last_processed}. Resetting progress.")
                    last_processed = 0
                    set_last_processed(state, gen_current_dataset_path, last_processed)

                next_needed = last_processed + GEN_CHUNK_SIZE
                if total_rows < next_needed or not is_low_usage():
                    print(f"[{datetime.now()}] Waiting... dataset={gen_current_dataset_path}, total={total_rows}, next_needed={next_needed}")
                    time.sleep(SLEEP_SECONDS_WHEN_IDLE)
                    continue

                start_idx = last_processed + 1
                end_idx   = start_idx + GEN_CHUNK_SIZE - 1
                print(f"[{datetime.now()}] Preparing chunk {start_idx}-{end_idx} / total={total_rows} for dataset={gen_current_dataset_path}")

                try:
                    chunk_ds = load_chunk_dataset(gen_current_dataset_path, start_idx, end_idx)
                except Exception as e:
                    print(f"[ERROR] Failed to select dataset rows {start_idx}-{end_idx} ({gen_current_dataset_path}): {e}")
                    time.sleep(SLEEP_SECONDS_WHEN_IDLE)
                    continue

                gen_run_counter[gen_current_project_tag] += 1
                run_dir = gen_current_project_work_root / f"run_{gen_run_counter[gen_current_project_tag]:03d}"
                run_dir.mkdir(parents=True, exist_ok=True)

                try:
                    delta_dir = gen_train_lora_on_chunk(gen_current_project_merged_dir, chunk_ds, run_dir)
                    gen_merge_delta_into_merged(delta_dir, gen_current_project_merged_dir)
                    # Route control after successful merge
                    write_route_control(
                        force_mode=os.environ.get("CTRL_FORCE_MODE", "student_forced"),
                        student_api_base=os.environ.get("MTLLM_STUDENT_API_BASE", "http://127.0.0.1:8010/v1"),
                        merged_dir=gen_current_project_merged_dir,
                        project_tag=gen_current_project_tag,
                        reason="merge_complete"
                    )
                except Exception as e:
                    print(f"[ERROR] Training/Merge failed for {gen_current_dataset_path}: {e}")
                    time.sleep(SLEEP_SECONDS_WHEN_IDLE)
                    continue

                set_last_processed(state, gen_current_dataset_path, end_idx)
                print(f"[{datetime.now()}] Completed chunk {start_idx}-{end_idx} for dataset={gen_current_dataset_path}. Updated merged: {gen_current_project_merged_dir}")

        except KeyboardInterrupt:
            print("\n[INFO] Stopping on user interrupt.")
            break
        except Exception as e:
            print(f"[FATAL] Loop error: {e}")
            time.sleep(SLEEP_SECONDS_WHEN_IDLE)

if __name__ == "__main__":
    main()
