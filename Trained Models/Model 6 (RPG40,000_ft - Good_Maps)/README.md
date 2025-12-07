# RPG Level Generator — LoRA Fine‑Tuned Model (40,000‑row dataset)

**Project goal:** Generate immersive, progressively challenging RPG maps as compact JSON/ASCII grids.  
**Model:** locally fine‑tuned causal LM on **40,000** curated level records, with a **GPT‑4o‑mini fallback** for extended depth.

---

## 🔗 Quick Links (replace with your Hugging Face URLs)

- **Model:** [Hugging Face model page](<https://huggingface.co/Hirudika2002/JARVIS-Models/tree/main/LoRA-Trained/Model-6-Stable(40%2C000_Rows)>)
- **Demo Video:** [Hugging Face Spaces / video](<https://huggingface.co/Hirudika2002/JARVIS-Models/blob/main/Model_Demo_Videos/Model_6(40000_DataLines)Demo_video.mp4>)
- **Dataset (used for training):** [Hugging Face dataset](<https://huggingface.co/Hirudika2002/JARVIS-Models/tree/main/DataSets/For%2040%2C000%20Rows>)

> Put these links at the very top so users can jump straight to the model, demo, and data.

---

## TL;DR

- Trained on **40k rows**, delivering **diverse, well‑structured** maps with smooth progression.  
- **Consistent quality from L2 → L10**, with **balanced difficulty** and **stable layouts**.  
- After **L10**, the system **seamlessly falls back** to **GPT‑4o‑mini** to sustain deeper levels without breaking immersion.  
- The **hybrid pipeline** (local fine‑tuned model + GPT fallback) enables scalable content generation.

---

## 📌 Performance

- Consistently generates **good‑quality**, **well‑structured** maps from **Level 2 up to Level 10**.  
- Layouts are **diverse, creative, and stable**, clearly improved vs. smaller datasets.  
- **Progression** feels smoother, with **natural transitions** and **balanced difficulty scaling**.  
- **Post‑L10**, the system transitions to **GPT‑4o‑mini** to support **extended gameplay**.  
- The **hybrid pipeline** ensures **scalable map generation** without breaking immersion.

---

## 🔎 Observations

Compared to 1k–2k row models, the 40k model shows:

- **More structured** map layouts and **better complexity handling**.  
- **Improved stability** across consecutive generations.  
- **Logical level progression**, supporting smoother gameplay.  
- Dataset scale contributes to **cohesive, playable** maps with stronger diversity.

---

## ⚠️ Limitations

- While **L2–L10** is reliably handled, **fallback is still required** for deeper levels.  
- Maps are **very good but not flawless**; occasional **imperfections** remain.  
- Further **scaling** and **additional fine‑tuning passes** may reduce reliance on fallback beyond L10.

---

## 🧱 Output Format & Invariants

All generations follow a compact JSON format:

```json
{
  "level": {"name": "7", "level_index": 7, "difficulty": 4, "time": 300, "width": 15, "height": 10},
  "walls": [{"start_pos": {"x": 0, "y": 0}, "end_pos": {"x": 14, "y": 0}}],
  "small_obstacles": [{"x": 6, "y": 3}, {"x": 8, "y": 6}],
  "enemies": [{"x": 4, "y": 5}, {"x": 10, "y": 7}],
  "player_pos": {"x": 1, "y": 1}
}
```

**Policy:** Use the **local model for L2–L10**; **fallback beyond L10** or on **validation failure**.

---

## 🧰 Training Notes

- **Dataset:** 40,000 rows; JSON schema with `level_index`, `difficulty`, and optional **complexity tags** (loops, keys/locks, branching).  
- **Curriculum:** mixed batches, with emphasis on **higher levels** to learn deeper patterns.  
- **Regularization:** weight decay (L2), occasional KL term to base model to reduce drift.  
- **PEFT:** LoRA support with rank sweeps (e.g., r ∈ {16, 32, 64}).  
- **Stop tokens:** explicit `END` marker to reduce truncation and early stops.


