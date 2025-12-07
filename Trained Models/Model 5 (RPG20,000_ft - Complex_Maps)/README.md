# RPG Level Generator — LoRA Fine‑Tuned Model (20,000‑row dataset)

**Project goal:** Generate richer, more complex RPG maps as compact JSON/ASCII grids.  
**Model:** locally fine‑tuned causal LM on **20,000** curated level records, with **GPT‑4o‑mini fallback** for late‑game depth (post L8–L10 as needed).

---

## 🔗 Quick Links (replace with your Hugging Face URLs)

- **Model:** [Hugging Face model page](<https://huggingface.co/Hirudika2002/JARVIS-Models/tree/main/LoRA-Trained/Model-5(20%2C000_Rows)>)
- **Demo Video:** [Hugging Face Spaces / video](<https://huggingface.co/Hirudika2002/JARVIS-Models/blob/main/Model_Demo_Videos/Model_5(20000_DataLines)Demo_video.mp4>)
- **Dataset (used for training):** [Hugging Face dataset](<https://huggingface.co/Hirudika2002/JARVIS-Models/tree/main/DataSets/For%2020%2C000%20Rows>)

> Keep these links at the very top so users can jump straight to the model, demo, and data.

---

## TL;DR

- Trained on **20k rows**; generates **higher structural quality** and **complexity** than smaller runs.  
- Smooth progression through **Levels 1 → 8**, with **occasional L9–L10** before fallback.  
- Layouts show **stronger patterns**, **better pathways**, and **more intricate designs**.  
- **Playable** outputs with **increased stability** across multiple levels.  
- **Retry loops** can appear due to **strict validator checks**; overall deeper maps still benefit from a **hybrid fallback**.

---

## 🎯 Performance Summary

| Capability | What you can expect |
|---|---|
| Level coverage | Stable L1–L8; sometimes L9–L10; fallback beyond 8–10 |
| Structure quality | Strong patterns, better pathways, more intricate designs |
| Stability | Improved across sequences; retries triggered by strict validation |
| Diversity | Much better vs. 10k/15k; more variation in pathways/layouts |
| Progression | Smoother, more coherent mid‑game progression |

---

## 🔎 Observations

- Larger dataset **significantly increased complexity** and reduced the simplicity seen in smaller runs.  
- **Structural diversity** is **notably better**, with more varied pathways and room/corridor motifs.  
- **Retries** occur during generation because of **stricter validators** (expected when targeting richer maps).  
- Dataset scale is **still insufficient** to cover the **deepest, highly complex** levels without assistance.

---

## ⚠️ Limitations

- Falls back after **Level 8–10**, limiting very deep progression on the local model alone.  
- **Retry loops** may increase latency when validators are overly strict.  
- Generalization beyond **mid‑to‑late** levels remains challenging.

---

**Policy:** Use the **local model for L1–L8** (sometimes L9–L10); **fallback** beyond **8–10** or on **validation failure/exhausted retries**.

**Validator suggestions:** Convert hard fails (e.g., duplicated obstacles) into **repairs** (dedupe or clamp) where feasible.

---

## 🧱 Output Format & Invariants

```json
{
  "level": {"name": "8", "level_index": 8, "difficulty": 5, "time": 300, "width": 15, "height": 10},
  "walls": [{"start_pos": {"x": 0, "y": 0}, "end_pos": {"x": 14, "y": 0}}],
  "small_obstacles": [{"x": 6, "y": 3}, {"x": 8, "y": 6}],
  "enemies": [{"x": 4, "y": 5}, {"x": 10, "y": 7}],
  "player_pos": {"x": 1, "y": 1}
}

