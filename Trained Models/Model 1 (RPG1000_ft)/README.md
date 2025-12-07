# RPG Level Generator — LoRA Fine‑Tuned Model (1000‑row dataset)

**Model type:** autoregressive text model fine‑tuned to emit compact JSON/ASCII tile maps for early‑game RPG levels.  
**Task:** procedural level generation (Level 1–3 focus).

---

## 🔗 Quick Links (replace with your Hugging Face URLs)

- **Model:** [Hugging Face model page](<https://huggingface.co/Hirudika2002/JARVIS-Models/tree/main/LoRA-Trained/Model-1(1000_Rows)>)

> Tip: keep these at the very top of the README so users can jump straight to the model, demo, and data.

---

## TL;DR

- Trained on **1,000 rows** to generate **RPG‑style maps**.  
- **Good** at **Level 1–2**; **occasionally** reaches **Level 3**.  
- **Stops early** fairly often; **rarely** reaches Level 4+.  
- Map structures are **playable but simple**.  
- Stability is **better** than tiny‑data runs, but still shows **early termination** and **repetition**.

---

## 🎯 Performance Summary

| Capability | What you can expect |
|---|---|
| Level coverage | Strong on L1–L2, sometimes L3, seldom deeper |
| Structure quality | Average; layouts are simple and functional |
| Diversity | Limited; some repetitive motifs |
| Stability | Improved vs. very small datasets; still ends early at times |
| Progression depth | Usually stops after Level 2 |

---

## 🔎 Observations

- Structures are **not substantially improved** vs. smaller training sets — overall design remains **average**.  
- Levels are **playable but minimal**, with **low puzzle/maze complexity**.  
- **Generation stability** improves over extremely small datasets, but **abrupt endings** still occur.

---

## ⚠️ Known Limitations

- **Average structure quality**, lacking creativity and detail.  
- **Fails to reach deeper levels** (4+).  
- Some outputs feel **repetitive** or **incomplete (early stop)**.

---

## 🧰 Data & Format

- **Dataset size:** 1,000 rows (training).  
- **Format:** JSON records representing levels and entities.  
- **Typical schema:**

```json
{
  "level": {
    "name": "1",
    "difficulty": 1,
    "time": 300,
    "width": 15,
    "height": 10
  },
  "walls": [
    {"start_pos": {"x": 0, "y": 0}, "end_pos": {"x": 0, "y": 9}}
  ],
  "small_obstacles": [{"x": 9, "y": 1}],
  "enemies": [{"x": 4, "y": 4}, {"x": 9, "y": 2}],
  "player_pos": {"x": 1, "y": 1}
}
```

> Keep **indices inside** `[0, width-1] × [0, height-1]`. Filtering invalid/duplicate coordinates at training time helps stability.


## 🧩 Example Output (format illustration)

```json
{
  "level": {"name": "2", "difficulty": 2, "time": 300, "width": 15, "height": 10},
  "walls": [
    {"start_pos": {"x": 0, "y": 0}, "end_pos": {"x": 0, "y": 9}},
    {"start_pos": {"x": 14, "y": 0}, "end_pos": {"x": 14, "y": 9}},
    {"start_pos": {"x": 2, "y": 4}, "end_pos": {"x": 12, "y": 4}}
  ],
  "small_obstacles": [{"x": 6, "y": 3}, {"x": 8, "y": 6}],
  "enemies": [{"x": 4, "y": 5}, {"x": 10, "y": 7}],
  "player_pos": {"x": 1, "y": 1}
}
```




