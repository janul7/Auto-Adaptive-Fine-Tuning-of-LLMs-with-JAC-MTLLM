# merge_peft.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os


BASE    = r"D:\Jac\TinyLlama-1.1B-Chat-v1.0"              # Base Model
ADAPTER = r"D:\Jac\FineTune_Test_7 (60,000_Rows)_Complex Maps"   # Your adapter folder (with adapter_model.safetensors)
OUT     = r"D:\Jac\TinyLlama-1.1B-Chat-v1.0-merged_7"     # New folder to save merged weights


os.makedirs(OUT, exist_ok=True)

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="auto", device_map="cpu")

model = PeftModel.from_pretrained(base, ADAPTER)
model = model.merge_and_unload()   

model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print("Merged model saved to:", OUT)
