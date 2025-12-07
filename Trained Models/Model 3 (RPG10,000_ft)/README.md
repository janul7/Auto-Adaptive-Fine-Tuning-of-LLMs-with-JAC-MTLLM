
# RPG Level Generator — LoRA Fine‑Tuned Model (10,000‑row dataset)

**Project goal:** Generate playable, progressively challenging RPG maps as compact JSON/ASCII grids.  
**Model:** locally fine‑tuned causal LM on **10,000** curated level records, with **GPT‑4o‑mini fallback** for post‑L6 depth.

---

## 🔗 Quick Links (replace with your Hugging Face URLs)

- **Model:** [Hugging Face model page](<https://huggingface.co/Hirudika2002/JARVIS-Models/tree/main/LoRA-Trained/Model-3(10%2C000_Rows)>)
- **Demo Video:** [Hugging Face Spaces / video](<https://huggingface.co/Hirudika2002/JARVIS-Models/blob/main/Model_Demo_Videos/Model_3(0000_DataLines)Demo_video.mp4>)


> Keep these links at the top so users can jump straight to the model, demo, and data.

---

## TL;DR

- Trained on **10k rows**; reliably generates **smooth progression up to Level 6** with **consistent structures**.  
- **Stability** is clearly improved over smaller datasets; **fewer abrupt terminations**.  
- **After Level 6**, generation **hands off to GPT‑4o‑mini** to extend gameplay.  
- Output maps are **playable** and **structured**, with improved **flow**; creativity/variety still **moderate**.

---

## 🎯 Performance Summary

| Capability | What you can expect |
|---|---|
| Level coverage | Stable L1–L6 locally; fallback after L6 |
| Structure quality | Playable, coherent layouts; average creativity |
| Stability | Fewer early stops vs. 1k/2k models |
| Variety | Moderate; motifs repeat at times |
| Progression | Smooth through L6, then GPT‑assisted beyond |

---

## 🔎 Observations

- Larger dataset **significantly improved reliability** and **coherence**.  
- **Complexity** within levels is **better** than smaller‑dataset runs, but **variety** remains **limited**.  
- Designs are **functional** rather than **innovative** — good for baseline playability.

---

## ⚠️ Limitations

- Lacks native support for **high‑level complexity** (L7+ handled via GPT fallback).  
- **Structural variety** is **moderate**; patterns can feel **repetitive**.  
- **Relies on GPT‑4o‑mini** for deeper progression.

---


**Policy:** Use the **local model for L1–L6**; **fallback** beyond **L6** or on **validation failure**.

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

- **Dataset:** 10,000 rows; JSON schema with `level_index`, `difficulty`, and optional **complexity tags**.  

---

