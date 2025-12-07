# RPG Level Generator — LoRA Fine‑Tuned Model (60,000‑row dataset)


**Base model:** TinyLLaMA‑1.1B (causal LM)  
**Goal:** Generate high‑quality, progressively challenging RPG maps as compact JSON/ASCII grids.  
**This is our best finetune so far**, trained on **60,000** curated level records, with a **GPT‑4o‑mini fallback** for late‑game depth.

---

## 🔗 Quick Links (replace with your Hugging Face URLs)

- **Model:** [Hugging Face model page](<https://huggingface.co/Hirudika2002/JARVIS-Models/tree/main/LoRA-Trained/Model-7-Advanced(60%2C000_Rows)>)
- **Demo Video:** [Hugging Face Spaces / video](<https://huggingface.co/Hirudika2002/JARVIS-Models/blob/main/Model_Demo_Videos/Model_7(60000_DataLines)Demo_video.mp4>)
- **Dataset (used for training):** [Hugging Face dataset](<https://huggingface.co/Hirudika2002/JARVIS-Models/tree/main/DataSets/For%2060%2C000%20Rows>)

> Keep these at the top so users can jump directly to the model, demo, and data.

---

## TL;DR

- **60k‑row** finetune on TinyLLaMA‑1.1B producing **complex, detailed** maps with **strong variety** and **stable progression**.  
- Consistently strong from **L1 → L8**; **often reaches L9–L10** before fallback.  
- **Quality is noticeably higher**: authentic feel, better pathways, advanced layouts.  
- Occasional **validator‑triggered retries**, but final outputs remain **stable and playable**.  
- Best balance so far between **stability, complexity, and quality**.

---

## 🎯 Performance Summary

| Capability | What you can expect |
|---|---|
| Level coverage | Consistent L1–L8; often L9–L10; fallback beyond 9–10 |
| Structure quality | High — advanced layouts, strong flow, improved pathways |
| Stability | High across multi‑level sequences; occasional retries due to strict validators |
| Diversity | Strong — reduced repetition vs. all earlier models |
| Progression | Smooth and engaging; difficulty scaling feels natural |

---

## 🔎 Observations

- The **60k dataset** captures **richer design patterns** and **reduces repetition**.  
- **Complexity and creativity** are **significantly better** than previous runs (10k–40k).  
- **Retries** during generation can occur because of **tighter validators**, which is expected when targeting complex maps.  
- Overall, this model achieves the **most balanced trade‑off** among **stability, complexity, and quality** to date.

---

## ⚠️ Limitations

- Still **falls back around L9–L10**; not sustaining very deep progression **natively**.  
- **Retry loops** may increase slightly for **very complex** layouts.  
- Even at 60k, the dataset may **not fully cover late‑game** structural diversity.

---


**Policy:** Use the **local TinyLLaMA‑1.1B finetune for L1–L8** (frequently L9–L10). Fall back **beyond L9–L10** or when **validation fails/exceeds retries**.


---

## 🧱 Output Format & Invariants

```json
{
  "level": {"name": "9", "level_index": 9, "difficulty": 5, "time": 300, "width": 15, "height": 10},
  "walls": [{"start_pos": {"x": 0, "y": 0}, "end_pos": {"x": 14, "y": 0}}],
  "small_obstacles": [{"x": 6, "y": 3}, {"x": 8, "y": 6}],
  "enemies": [{"x": 4, "y": 5}, {"x": 10, "y": 7}],
  "player_pos": {"x": 1, "y": 1}
}


