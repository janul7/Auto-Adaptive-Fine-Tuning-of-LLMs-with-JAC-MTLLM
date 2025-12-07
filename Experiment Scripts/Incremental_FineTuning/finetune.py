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
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from json import JSONDecodeError

import gc
import stat
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

class FP32TrainingArguments(TrainingArguments):
    def __post_init__(self):
        self.fp16 = False
        self.bf16 = False
        super().__post_init__()


INITIAL_BASE_MODEL = os.environ.get("INITIAL_BASE_MODEL", r"D:\Jac\TinyLlama-1.1B-Chat-v1.0")

WORK_ROOT   = Path(os.environ.get("WORK_ROOT",   r"D:\Jaseci\finetune_runs")).resolve()
MERGED_ROOT = Path(os.environ.get("MERGED_ROOT", str(WORK_ROOT.parent / "merged_models"))).resolve()

CHUNK_SIZE   = int(os.environ.get("CHUNK_SIZE", "2"))
MAX_SEQ_LEN  = int(os.environ.get("MAX_SEQ_LEN", "512"))
NUM_EPOCHS   = int(os.environ.get("NUM_EPOCHS", "1"))
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", "1"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-3"))
SLEEP_SECONDS_WHEN_IDLE = int(os.environ.get("SLEEP_SECONDS_WHEN_IDLE", "30"))

def is_low_usage() -> bool:
    return True


POINTER_PATH = Path(os.environ.get("ACTIVE_LOG_POINTER", str(Path.home() / ".jaseci" / "active_log.json")))

WORK_ROOT.mkdir(parents=True, exist_ok=True)
MERGED_ROOT.mkdir(parents=True, exist_ok=True)
STATE_PATH = WORK_ROOT / "state.json"

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

def save_state(state: Dict[str, Any]) -> None:
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def get_last_processed(state: Dict[str, Any], dataset_path: str) -> int:
    return int(state["datasets"].get(dataset_path, {}).get("last_processed", 0))

def set_last_processed(state: Dict[str, Any], dataset_path: str, idx: int) -> None:
    state["datasets"].setdefault(dataset_path, {})
    state["datasets"][dataset_path]["last_processed"] = int(idx)
    save_state(state)


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]

def project_paths_for_dataset(dataset_jsonl: Path) -> Tuple[str, Path, Path]:
    project_dir = dataset_jsonl.resolve().parent
    tag = f"{project_dir.name}__{_short_hash(str(project_dir))}"
    project_work_root = WORK_ROOT / tag
    project_merged_dir = MERGED_ROOT / tag / "merged"
    project_work_root.mkdir(parents=True, exist_ok=True)
    project_merged_dir.parent.mkdir(parents=True, exist_ok=True)
    return tag, project_work_root, project_merged_dir


def get_dataset_len(dataset_jsonl_path: str) -> int:
    ds = load_dataset("json", data_files=dataset_jsonl_path, split="train")
    return len(ds)

def load_chunk_dataset(dataset_jsonl_path: str, start_idx: int, end_idx: int):
    ds = load_dataset("json", data_files=dataset_jsonl_path, split="train")
    indices = list(range(start_idx - 1, end_idx))
    return ds.select(indices)

def build_text(sample: Dict[str, Any]) -> str:
    system = (sample.get("system") or "").strip()
    user_in = (sample.get("input") or "").strip()
    out     = (sample.get("output") or "").strip()
    return (system + "\n" + user_in + "\n" + out).strip()

def preprocess_dataset(ds, tokenizer):
    def _map_fn(sample):
        text = build_text(sample)
        tokenized = tokenizer(
            text,
            max_length=MAX_SEQ_LEN,
            truncation=True,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    cols = ds.column_names
    return ds.map(_map_fn, remove_columns=cols)


def ensure_merged_exists(merged_dir: Path):
    """Recreate merged_dir from INITIAL_BASE_MODEL if missing/empty."""
    if merged_dir.exists() and any(merged_dir.iterdir()):
        return
    print(f"[{datetime.now()}] Initializing merged model at: {merged_dir}")
    if merged_dir.exists():
        
        shutil.rmtree(merged_dir, ignore_errors=True)
        time.sleep(0.2)
    shutil.copytree(INITIAL_BASE_MODEL, merged_dir)


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
    for i in range(retries):
        try:
            shutil.rmtree(path, onerror=_on_rm_error)
            return
        except Exception:
            time.sleep(delay)
  
    shutil.rmtree(path, ignore_errors=True)

def _atomic_replace_dir(src: Path, dst: Path, retries: int = 8, delay: float = 0.4) -> None:
    """
    Replace dst with src (directories). Retries to avoid Windows file locks.
    """
    if dst.exists():
        _rmtree_retry(dst, retries=retries, delay=delay)
        time.sleep(0.15)

    for i in range(retries):
        try:
            os.replace(str(src), str(dst))
            return
        except PermissionError:
            time.sleep(delay)
        except OSError:
            time.sleep(delay)

    
    if not dst.exists():
        shutil.copytree(str(src), str(dst))
    shutil.rmtree(str(src), ignore_errors=True)


def train_lora_on_chunk(base_model_dir: Path, chunk_ds, run_dir: Path) -> Path:
    """
    Train LoRA on chunk starting from base_model_dir (this project's merged model).
    Returns the LoRA delta dir.
    """
    
    ensure_merged_exists(base_model_dir)

    base_posix = base_model_dir.as_posix() 
    print(f"[{datetime.now()}] Loading tokenizer/model from: {base_model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(base_posix, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_posix,
        device_map="cpu",
        torch_dtype=torch.float32,  
        low_cpu_mem_usage=False,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj"],
        bias="none",
    )
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

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)

    print(f"[{datetime.now()}] Starting LoRA training on chunk...")
    trainer.train()

    delta_dir = run_dir / "delta"
    delta_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{datetime.now()}] Saving LoRA delta to: {delta_dir}")
    trainer.model.save_pretrained(str(delta_dir))
    tokenizer.save_pretrained(str(delta_dir))

    
    del trainer, model, tokenizer
    gc.collect()

    
    latest = run_dir.parent / "adapter_latest"
    if latest.exists():
        _rmtree_retry(latest)
    shutil.copytree(str(delta_dir), str(latest))

    return delta_dir


def merge_delta_into_merged(delta_dir: Path, merged_dir: Path):
    """
    Merge the LoRA delta into merged_dir in place (Windows-safe).
    """
    print(f"[{datetime.now()}] Merging delta into merged model at: {merged_dir}")

    merged_posix = merged_dir.as_posix()
    delta_posix = delta_dir.as_posix()

    
    tok = AutoTokenizer.from_pretrained(merged_posix, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        merged_posix,
        device_map="cpu",            
        torch_dtype=torch.float16,   
    )

    model = PeftModel.from_pretrained(base, delta_posix)
    model = model.merge_and_unload()
    model = model.to(torch.float16)

    parent = merged_dir.parent
    tmp_out = Path(tempfile.mkdtemp(prefix="merge_tmp_", dir=str(parent)))

    model.save_pretrained(tmp_out.as_posix())
    tok.save_pretrained(tmp_out.as_posix())

    
    del model, base, tok
    gc.collect()
    time.sleep(0.2)

    
    _atomic_replace_dir(tmp_out, merged_dir)

    print(f"[{datetime.now()}] Merge complete. Updated merged model at: {merged_dir}")


ROUTE_CTRL_PATH = Path(os.environ.get(
    "LLM_ROUTE_CTRL",
    str(Path.home() / ".jaseci" / "llm_route.json")
))

def write_route_control(force_mode: str = "student_forced",
                        student_api_base: Optional[str] = None,
                        merged_dir: Optional[Path] = None,
                        project_tag: Optional[str] = None,
                        reason: str = "merge_complete"):
    """
    Atomically write a small JSON that the plugin will read on the next call.
    """
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


def main():
    state = load_state()

    current_dataset_path: Optional[str] = None
    current_project_tag: Optional[str] = None
    current_project_work_root: Optional[Path] = None
    current_project_merged_dir: Optional[Path] = None

    per_project_version: Dict[str, int] = {}

    while True:
        try:
            pointer = read_active_dataset_from_pointer()
            if not pointer:
                print(f"[{datetime.now()}] Waiting for pointer at: {POINTER_PATH}")
                time.sleep(SLEEP_SECONDS_WHEN_IDLE)
                continue

            dataset_path = str(Path(pointer["dataset_jsonl"]).resolve())
            if current_dataset_path != dataset_path:
                tag, proj_work, proj_merged = project_paths_for_dataset(Path(dataset_path))
                current_dataset_path = dataset_path
                current_project_tag = tag
                current_project_work_root = proj_work
                current_project_merged_dir = proj_merged
                per_project_version.setdefault(tag, 0)
                print(f"[{datetime.now()}] Switching active dataset to: {dataset_path}")
                print(f"[{datetime.now()}] Project tag: {tag}")
                ensure_merged_exists(current_project_merged_dir)

            total_rows = get_dataset_len(current_dataset_path)
            last_processed = get_last_processed(state, current_dataset_path)

            if total_rows < last_processed:
                print(f"[{datetime.now()}] Detected dataset reset/rotation. total={total_rows} < last={last_processed}. Resetting progress.")
                last_processed = 0
                set_last_processed(state, current_dataset_path, last_processed)

            next_needed = last_processed + CHUNK_SIZE
            if total_rows < next_needed or not is_low_usage():
                print(f"[{datetime.now()}] Waiting... dataset={current_dataset_path}, total={total_rows}, next_needed={next_needed}")
                time.sleep(SLEEP_SECONDS_WHEN_IDLE)
                continue

            start_idx = last_processed + 1
            end_idx   = start_idx + CHUNK_SIZE - 1
            print(f"[{datetime.now()}] Preparing chunk {start_idx}-{end_idx} / total={total_rows} for dataset={current_dataset_path}")

            try:
                chunk_ds = load_chunk_dataset(current_dataset_path, start_idx, end_idx)
            except Exception as e:
                print(f"[ERROR] Failed to select dataset rows {start_idx}-{end_idx} ({current_dataset_path}): {e}")
                time.sleep(SLEEP_SECONDS_WHEN_IDLE)
                continue

            per_project_version[current_project_tag] += 1
            run_dir = current_project_work_root / f"run_{per_project_version[current_project_tag]:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            try:
                delta_dir = train_lora_on_chunk(current_project_merged_dir, chunk_ds, run_dir)
                merge_delta_into_merged(delta_dir, current_project_merged_dir)
            except Exception as e:
                print(f"[ERROR] Training/Merge failed for {current_dataset_path}: {e}")
                time.sleep(SLEEP_SECONDS_WHEN_IDLE)
                continue

            
            write_route_control(
                force_mode=os.environ.get("CTRL_FORCE_MODE", "student_forced"),  
                student_api_base=os.environ.get("MTLLM_STUDENT_API_BASE", "http://127.0.0.1:8010/v1"),
                merged_dir=current_project_merged_dir,
                project_tag=current_project_tag,
                reason="merge_complete"
            )

            set_last_processed(state, current_dataset_path, end_idx)
            print(f"[{datetime.now()}] Completed chunk {start_idx}-{end_idx} for dataset={current_dataset_path}. Updated merged: {current_project_merged_dir}")

        except KeyboardInterrupt:
            print("\n[INFO] Stopping on user interrupt.")
            break
        except Exception as e:
            print(f"[FATAL] Loop error: {e}")
            time.sleep(SLEEP_SECONDS_WHEN_IDLE)

if __name__ == "__main__":
    main()
