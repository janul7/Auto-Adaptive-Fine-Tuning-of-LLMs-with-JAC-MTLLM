# RPG Level Generator — LoRA Fine‑Tuned Model (2000‑row dataset)

**Model type:** autoregressive text model fine‑tuned to emit compact JSON/ASCII tile maps for mid‑early RPG levels.  
**This is the 2nd finetune**, trained on a **2,000‑row** dataset.

---

## 🔗 Quick Links (replace with your Hugging Face URLs)

- **Model:** [Hugging Face model page](<https://huggingface.co/Hirudika2002/JARVIS-Models/tree/main/LoRA-Trained/Model-2(2000_Rows)>)


> Keep these links at the top so users can jump straight to the model, demo, and data.

---

## TL;DR

- Trained on **2,000 rows** to generate **RPG‑style maps**.  
- Reliably outputs **Levels 2–6 patterns**, but **complexity does not consistently increase** with level number.  
- **Stability improved** over the 1k run; **differentiation** and **variety** remain limited.  
- Some **structural errors** can appear after several maps in a single generation session.

---

## 🎯 Performance Summary

| Capability | What you can expect |
|---|---|
| Level coverage | Consistent patterns for L2–L6; early levels (L1–L2) remain very stable |
| Structure quality | **Average** — functional, playable, but not very rich |
| Diversity | **Moderate‑low** — motifs repeat; limited layout variety |
| Progression | **Weak** — complexity does not clearly ramp with level index |
| Robustness | Improved vs. 1k model; occasional structural inconsistencies on long runs |

---

## 🔎 Observations

- Increasing the dataset from **1k → 2k** improved **stability** (more levels generate correctly before error).  
- **Level‑to‑level differentiation** is still limited — designs feel **repetitive** and **homogeneous**.  
- After several consecutive generations, **structural errors** or **inconsistencies** (e.g., dangling walls, invalid coordinates) can surface.

---

## ⚠️ Known Limitations

- **Average structure quality**, lacking novelty and layered objectives.  
- **Progression gap:** designs for L3–L6 often resemble L2 in density/maze depth.  
- **Repetition:** corridor and room motifs recur; some incomplete maps on long runs.

---

## 🚀 Recommended Next Steps

1. **Scale & diversify the dataset**  
   - Grow beyond **2k** (e.g., **5k–10k**) with **richer templates** and **varied objectives** (keys/locks, loops, secret rooms).  
   - Balance examples across **level indices** with explicit signals about intended **complexity** (enemies, branching factor, puzzle count).

2. **Regularization for stability**  
   - **L2 / weight decay** (`0.01–0.05` depending on optimizer & base model).  
   - **KL‑regularization** to the base model (policy‑KL style) to discourage drift/collapse.  
   - **Label smoothing** for minor robustness gains.

3. **Curriculum learning** (enforce increasing complexity)  
   - Stage A: L1–L2 simple layouts → Stage B: L3–L4 with more branching → Stage C: L5–L6 with constraints (dead‑ends allowed, loops/keys).  
   - Use **level index** and **complexity tags** in prompts; gradually expand allowed motif set.

4. **LoRA rank search**  
   - Sweep **r ∈ {8, 16, 32, 64}** and compare on diversity/valid‑JSON rate and progression metrics.  
   - Consider **layer‑wise rank** (higher r on attention blocks that handle long‑range structure).

5. **Decoding controls & guards**  
   - Use `min_new_tokens` / `min_length` to mitigate early stops.  
   - `top_p ≈ 0.9–0.95`, `temperature ≈ 0.8–1.0`, `repetition_penalty ≈ 1.05–1.15`, `no_repeat_ngram_size = 3–4`.  
   - Add **stop sequences** and perform **streaming JSON validation**; run a **repair pass** if small violations occur.

6. **Constraint‑aware/guided decoding**  
   - Validate and clamp coordinates to `[0, width-1] × [0, height-1]`.  
   - Optionally integrate a **structure‑checking function** (connectedness, border seals, enemy spacing) to nudge decoding.

---

## 🧰 Data & Format

- **Dataset size:** 2,000 rows (training).  
- **Format:** JSON records representing levels and entities.  
- **Schema signal:** include **`level_index`/`difficulty`** and **complexity tags** to help the model learn progression.

```json
{
  "level": {
    "name": "5",
    "level_index": 5,
    "difficulty": 3,
    "time": 300,
    "width": 15,
    "height": 10,
    "tags": ["loops", "branching", "keys"]
  },
  "walls": [
    {"start_pos": {"x": 0, "y": 0}, "end_pos": {"x": 0, "y": 9}},
    {"start_pos": {"x": 14, "y": 0}, "end_pos": {"x": 14, "y": 9}}
  ],
  "small_obstacles": [{"x": 6, "y": 3}, {"x": 8, "y": 6}],
  "enemies": [{"x": 4, "y": 5}, {"x": 10, "y": 7}],
  "player_pos": {"x": 1, "y": 1}
}
```

## 🧩 Example Output (format illustration)

```json
{
  "level": {"name": "4", "level_index": 4, "difficulty": 3, "time": 300, "width": 15, "height": 10},
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


