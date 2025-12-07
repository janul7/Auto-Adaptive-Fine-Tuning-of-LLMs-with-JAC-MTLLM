# RPG Level Generator — LoRA Fine‑Tuned Model (15,000‑row dataset)

**Project goal:** Generate playable, progressively challenging RPG maps as compact JSON/ASCII grids.  
**Model:** locally fine‑tuned causal LM on **15,000** curated level records, with **GPT‑4o‑mini fallback** for extended depth beyond L5–L6.

---

## 🔗 Quick Links (replace with your Hugging Face URLs)

- **Model:** [Hugging Face model page](<https://huggingface.co/Hirudika2002/JARVIS-Models/tree/main/LoRA-Trained/Model-4(15%2C000_Rows)>)
- **Demo Video:** [Hugging Face Spaces / video](<https://huggingface.co/Hirudika2002/JARVIS-Models/blob/main/Model_Demo_Videos/Model_4(15000_DataLines)Demo_video.mp4>)


> Keep these links at the top so users can jump straight to the model, demo, and data.

---

## TL;DR

- Trained on **15k rows**, producing **stable, consistent** maps up to **Level 5**, with **occasional Level 6**.  
- **Slightly improved** structure and **creativity** vs. the **10k** model; levels feel **more RPG‑like**.  
- **Progression** is **smooth** and **coherent** through L5; deeper levels remain challenging.  
- **Fallback to GPT‑4o‑mini** is used around **L5–L6** for extended progression.

---

## 🎯 Performance Summary

| Capability | What you can expect |
|---|---|
| Level coverage | Stable L1–L5 locally; sometimes L6; fallback after L5–L6 |
| Structure quality | Improved vs. 10k; better structure and moderate creativity |
| Stability | Strong through L5; fewer abrupt terminations |
| Variety | Better than 10k, but some repetition persists |
| Progression | Smooth and coherent through L5; deeper levels are harder |

---

## 🔎 Observations

- Larger dataset **improves diversity** and **reduces repetition**.  
- Levels are **more “RPG‑like”** with better **structural flow** and **consistency**.  
- **Progression stability** is strong until **Level 5**; deeper levels are still challenging.

---

## ⚠️ Limitations

- Still falls back around **Level 5–6**, not sustaining higher‑level generation natively.  
- **Creativity** improved, but not dramatically — maps are **functional**, not highly **innovative**.  
- **Some structural repetition** remains despite dataset growth.

---


**Policy:** Use the **local model for L1–L5 (sometimes L6)**; **fallback** beyond **L5–L6** or on **validation failure**.

---

## 🧱 Output Format & Invariants

```json
{
  "level": {"name": "5", "level_index": 5, "difficulty": 3, "time": 300, "width": 15, "height": 10},
  "walls": [{"start_pos": {"x": 0, "y": 0}, "end_pos": {"x": 14, "y": 0}}],
  "small_obstacles": [{"x": 6, "y": 3}, {"x": 8, "y": 6}],
  "enemies": [{"x": 4, "y": 5}, {"x": 10, "y": 7}],
  "player_pos": {"x": 1, "y": 1}
}
```

## 🧰 Data 

- **Dataset:** 15,000 rows; JSON schema with `level_index`, `difficulty`, and optional **complexity tags**.  


