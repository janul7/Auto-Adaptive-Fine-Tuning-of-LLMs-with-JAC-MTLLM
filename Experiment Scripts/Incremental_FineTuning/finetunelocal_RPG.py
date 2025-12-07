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


try:
    from pattern_validator import validate_text
except Exception as _e:
    validate_text = None
    print(f"[WARN] pattern_validator not importable: {_e}")


class FP32TrainingArguments(TrainingArguments):
    def __post_init__(self):
        self.fp16 = False
        self.bf16 = False
        super().__post_init__()


INITIAL_BASE_MODEL = os.environ.get("INITIAL_BASE_MODEL", r"D:\Jac\TinyLlama-1.1B-Chat-v1.0")


WORK_ROOT   = Path(os.environ.get("WORK_ROOT",   r"D:\Jaseci\finetune_runs")).resolve()
MERGED_ROOT = Path(os.environ.get("MERGED_ROOT", str(WORK_ROOT.parent / "merged_models"))).resolve()


CHUNK_SIZE   = int(os.environ.get("CHUNK_SIZE", "10"))  
MAX_SEQ_LEN  = int(os.environ.get("MAX_SEQ_LEN", "512"))
NUM_EPOCHS   = int(os.environ.get("NUM_EPOCHS", "1"))
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", "1"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-3"))
SLEEP_SECONDS_WHEN_IDLE = int(os.environ.get("SLEEP_SECONDS_WHEN_IDLE", "30"))


MILESTONE_ROWS = int(os.environ.get("MILESTONE_ROWS", "10"))           
VALIDATION_SAMPLE_ROWS = int(os.environ.get("VALIDATION_SAMPLE_ROWS", "4"))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "1337"))
random.seed(RANDOM_SEED)


MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "2000"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))              
TOP_P = float(os.environ.get("TOP_P", "1.0"))
DO_SAMPLE = bool(int(os.environ.get("DO_SAMPLE", "0")))                

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



def _pf(p: Path) -> str:
    """Path -> resolved POSIX string (Windows-safe for HF loaders)."""
    return p.resolve().as_posix()

def _debug_list_dir(p: Path, label: str):
    try:
        items = sorted(x.name for x in p.iterdir())
        print(f"[{datetime.now()}] {label} exists={p.exists()} files={items[:12]}")
    except Exception as e:
        print(f"[{datetime.now()}] {label} list failed: {e}")



_RPG_RUN_WHITELIST = [
    r"D:\Jaseci\jaseci\jac\examples\rpg_game\jac_impl\jac_impl_6\main.jac",
]

def _normcasepath(p: str) -> str:
    return os.path.normcase(os.path.normpath(p or ""))

def _looks_like_rpg(run_file: str, dataset_path: str) -> bool:
    rf = _normcasepath(run_file)
    for w in _RPG_RUN_WHITELIST:
        wn = _normcasepath(w)
        if rf == wn:
            return True
        if rf.startswith(os.path.dirname(wn)):
            return True
    dp = _normcasepath(dataset_path)
    return "rpg_game" in dp


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]

def project_paths_for_dataset(dataset_jsonl: Path) -> Tuple[str, Path, Path]:
    """
    Given dataset path .../<project>/llm_io_log.jsonl
    Returns (project_tag, project_work_root, project_merged_dir)
    """
    project_dir = dataset_jsonl.resolve().parent
    tag = f"{project_dir.name}{_short_hash(str(project_dir))}"
    project_work_root = WORK_ROOT / tag
    project_merged_dir = MERGED_ROOT / tag / "merged"
    project_work_root.mkdir(parents=True, exist_ok=True)
    project_merged_dir.parent.mkdir(parents=True, exist_ok=True)
    return tag, project_work_root, project_merged_dir



def get_dataset_len(dataset_jsonl_path: str) -> int:
    ds = load_dataset("json", data_files=dataset_jsonl_path, split="train")
    return len(ds)

def load_dataset_first_n(dataset_jsonl_path: str, n: int):
    ds = load_dataset("json", data_files=dataset_jsonl_path, split="train")
    n = min(n, len(ds))
    indices = list(range(n))  
    return ds.select(indices)

def build_text(sample: Dict[str, Any]) -> str:
    system = (sample.get("system") or "").strip()
    user_in = (sample.get("input") or "").strip()
    out     = (sample.get("output") or "").strip()
    return (system + "\n" + user_in + "\n" + out).strip()

def build_prompt_from_row(row: Dict[str, Any]) -> str:
    """Construct the inference prompt from system + input ONLY."""
    system = (row.get("system") or "").strip()
    user_in = (row.get("input") or "").strip()
    return f"{system}\n{user_in}\n"

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
    """
    Ensure merged_dir is a valid HF model directory.
    If missing or invalid, (re)initialize by copying INITIAL_BASE_MODEL.
    """
    def _is_valid_model_dir(p: Path) -> bool:
        has_config = (p / "config.json").exists()
        has_weights = (p / "pytorch_model.bin").exists() or (p / "model.safetensors").exists()
        return p.exists() and has_config and has_weights

    if _is_valid_model_dir(merged_dir):
        return  

    print(f"[{datetime.now()}] Initializing merged model at: {merged_dir}")
    if merged_dir.exists():
        shutil.rmtree(merged_dir, ignore_errors=True)

    src = Path(INITIAL_BASE_MODEL)
    if not _is_valid_model_dir(src):
        raise RuntimeError(
            f"INITIAL_BASE_MODEL does not look like a valid HF model dir: {src}\n"
            f"Expected at least config.json and model weights (pytorch_model.bin or model.safetensors)."
        )

    merged_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, merged_dir)


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

def train_lora_on_dataset(base_model_dir: Path, ds, run_dir: Path) -> Path:
    """
    Train LoRA on the provided dataset starting from base_model_dir (this project's merged model).
    Returns the LoRA delta dir.
    """
    
    try:
        base_model_dir.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    ensure_merged_exists(base_model_dir)

    _debug_list_dir(base_model_dir, "Base/Merged model dir before load")

    print(f"[{datetime.now()}] Loading tokenizer/model from: {base_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(_pf(base_model_dir), use_fast=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        _pf(base_model_dir),
        device_map="cpu",
        torch_dtype=torch.float32,   
        low_cpu_mem_usage=False,
        local_files_only=True,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    
    ref_model = AutoModelForCausalLM.from_pretrained(
        _pf(base_model_dir),
        device_map="cpu",
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj"],
        bias="none",
    )
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

    trainer = RegularizedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        ref_model=ref_model,
        reg_lambda=0.01
    )

    print(f"[{datetime.now()}] Starting LoRA training on dataset (rows={len(ds)}) ...")
    trainer.train()

    delta_dir = run_dir / "delta"
    delta_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{datetime.now()}] Saving LoRA delta to: {delta_dir}")
    trainer.model.save_pretrained(str(delta_dir))
    tokenizer.save_pretrained(str(delta_dir))

    
    latest = run_dir.parent / "adapter_latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(str(delta_dir), str(latest))

    return delta_dir


def merge_delta_into_merged(delta_dir: Path, merged_dir: Path):
    """
    Merge the LoRA delta into merged_dir in place (via temp dir swap).
    """
    print(f"[{datetime.now()}] Merging delta into merged model at: {merged_dir}")

    tok = AutoTokenizer.from_pretrained(_pf(merged_dir), use_fast=True, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        _pf(merged_dir),
        device_map="cpu",            
        torch_dtype=torch.float16,   
        local_files_only=True,
    )

    model = PeftModel.from_pretrained(base, str(delta_dir))
    model = model.merge_and_unload()
    model = model.to(torch.float16)

    parent = merged_dir.parent
    tmp_out = Path(tempfile.mkdtemp(prefix="merge_tmp_", dir=str(parent)))
    model.save_pretrained(str(tmp_out))
    tok.save_pretrained(str(tmp_out))

    
    if merged_dir.exists():
        shutil.rmtree(merged_dir, ignore_errors=True)
    os.replace(str(tmp_out), str(merged_dir))

    print(f"[{datetime.now()}] Merge complete. Updated merged model at: {merged_dir}")



def _read_jsonl_first_n(dataset_path: Path, n: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                obj = {}
            out.append(obj)
    return out

def _load_model_with_delta_for_infer(base_model_dir: Path, delta_dir: Path):
    """
    Load tokenizer and base model, then attach LoRA delta for inference (no merge).
    """
    print(f"[GATE][INFER] Loading merged base for inference: {base_model_dir}")
    print(f"[GATE][INFER] Attaching fresh adapter (not merged): {delta_dir}")

    tokenizer = AutoTokenizer.from_pretrained(_pf(base_model_dir), use_fast=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    base = AutoModelForCausalLM.from_pretrained(
        _pf(base_model_dir),
        device_map=None,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
        local_files_only=True,
    ).to(device)

    model = PeftModel.from_pretrained(base, str(delta_dir))
    model.eval()
    return tokenizer, model, device

def _generate_single(model, tokenizer, device, prompt: str) -> str:
    """Generate text continuation for a single prompt; return only the new text."""
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

def validator_gate_inference(dataset_path: Path, milestone_rows: int, sample_k: int,
                             base_model_dir: Path, delta_dir: Path) -> bool:
    """
    Randomly sample K rows from the first 'milestone_rows' rows, build prompts
    from ("system","input"), run INFERENCE using (merged_base + loRA delta),
    then validate the generated outputs via pattern_validator.
    Returns True iff ALL sampled generations are valid (LEVEL or MAP).
    """
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


def compute_next_milestone(total_rows: int, last_processed: int, unit: int) -> Optional[int]:
    """
    Returns the largest multiple of 'unit' not yet processed and <= total_rows.
    Example: total_rows=23, unit=10, last_processed=10 -> returns 20.
    If nothing new to process, returns None.
    """
    if total_rows < unit:
        return None
    candidate = (total_rows // unit) * unit
    if candidate > last_processed:
        return candidate
    return None


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
            run_file = pointer.get("run_file") or ""

            
            if not _looks_like_rpg(run_file, dataset_path):
                print(f"[DISPATCH] Non-RPG dataset active → RPG trainer idling. run_file={run_file!r}")
                time.sleep(SLEEP_SECONDS_WHEN_IDLE)
                continue
            

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

            
            milestone = compute_next_milestone(total_rows, last_processed, MILESTONE_ROWS)
            if milestone is None or not is_low_usage():
                print(f"[{datetime.now()}] Waiting... dataset={current_dataset_path}, total={total_rows}, last_processed={last_processed}, unit={MILESTONE_ROWS}")
                time.sleep(SLEEP_SECONDS_WHEN_IDLE)
                continue

            print(f"[{datetime.now()}] Preparing milestone {milestone} (train on first {milestone} rows) / total={total_rows} for dataset={current_dataset_path}")

            
            current_project_merged_dir.parent.mkdir(parents=True, exist_ok=True)
            ensure_merged_exists(current_project_merged_dir)

            
            try:
                ds = load_dataset_first_n(current_dataset_path, milestone)
            except Exception as e:
                print(f"[ERROR] Failed to load first {milestone} rows from {current_dataset_path}: {e}")
                time.sleep(SLEEP_SECONDS_WHEN_IDLE)
                continue

            
            per_project_version[current_project_tag] += 1
            run_dir = current_project_work_root / f"run_{per_project_version[current_project_tag]:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            
            try:
                delta_dir = train_lora_on_dataset(current_project_merged_dir, ds, run_dir)
            except Exception as e:
                print(f"[ERROR] Training failed for {current_dataset_path}: {e}")
                time.sleep(SLEEP_SECONDS_WHEN_IDLE)
                continue

            
            try:
                gate_ok = validator_gate_inference(
                    Path(current_dataset_path),
                    milestone,
                    VALIDATION_SAMPLE_ROWS,
                    current_project_merged_dir,   
                    delta_dir                      
                )
            except Exception as e:
                print(f"[ERROR] Validator gate crashed: {e}")
                gate_ok = False

            if gate_ok:
                try:
                    merge_delta_into_merged(delta_dir, current_project_merged_dir)
                    print(f"[{datetime.now()}] Milestone {milestone}: merge SUCCESS.")
                except Exception as e:
                    print(f"[ERROR] Merge failed for milestone {milestone}: {e}")
                    time.sleep(SLEEP_SECONDS_WHEN_IDLE)
                    continue
            else:
                print(f"[{datetime.now()}] Milestone {milestone}: validator gate FAILED. Skipping merge; adapters kept on disk only.")

            
            set_last_processed(state, current_dataset_path, milestone)
            print(f"[{datetime.now()}] Completed milestone {milestone} for dataset={current_dataset_path}. Current merged: {current_project_merged_dir}")

        except KeyboardInterrupt:
            print("\n[INFO] Stopping on user interrupt.")
            break
        except Exception as e:
            print(f"[FATAL] Loop error: {e}")
            time.sleep(SLEEP_SECONDS_WHEN_IDLE)

if __name__ == "__main__":
    main()
