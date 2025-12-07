# finetune_tinyllama.py

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import torch
import os


# 1. Load Model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",          # auto-assign GPU if available
    torch_dtype=torch.float16
)


# 2. Load Dataset
print("Loading dataset...")
raw_data = load_dataset("json", data_files="new_dataset2(lvl1 and 2 500).jsonl")

print("Example row from dataset:")
print(raw_data["train"][0])


# 3. Preprocess Data
def preprocess(sample):
    text = sample["system"] + "\n" + sample["input"] + "\n" + sample["output"]

    tokenized = tokenizer(
        text,
        max_length=2048, ## too long for limited GPU memory — reduce (e.g., 512/1024) if OOM errors occur
        truncation=True,
        padding="max_length",
    )

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

data = raw_data.map(preprocess, remove_columns=raw_data["train"].column_names)

print("Tokenized sample:")
print(data["train"][0])


# 4. Apply LoRA
print("Applying LoRA configuration...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)


# 5. Training Setup
training_args = TrainingArguments(
    output_dir="./tinyllama-finetune",   # directory for logs & checkpoints
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    learning_rate=0.001,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=10,
    report_to="none",          # disable WandB/Hub logging
    save_strategy="epoch"      # save every epoch
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data["train"]
)


# 6. Train
print("Starting training...")
trainer.train()


# 7. Save Model Locally
model_path = "./finetuned-tinyllama"
os.makedirs(model_path, exist_ok=True)

print(f"Saving fine-tuned model to {model_path} ...")
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

