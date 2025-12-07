RPG Level Generator – Fine-Tuned Model (40,000 Rows)

This project introduces a fine-tuned model trained on 40,000 rows of RPG-style map data to generate immersive and progressively challenging game levels.

📌 Performance

Consistently generates good-quality, well-structured maps from Level 2 up to Level 10.

Layouts are diverse, creative, and stable, showing noticeable improvements compared to smaller dataset runs.

Progression feels smoother, with natural level transitions and balanced difficulty scaling.

After Level 10, the system seamlessly transitions to the GPT-4o-mini model to support extended gameplay.

The hybrid pipeline (fine-tuned local model + GPT fallback) ensures scalable map generation without breaking immersion.

🔎 Observations

Compared to smaller dataset models (1,000–2,000 rows), this model demonstrates:

More structured map layouts.

Better complexity handling.

Improved stability across generations.

Levels progress logically, supporting smoother gameplay experiences.

The dataset scale clearly contributes to more cohesive and playable maps.

⚠️ Limitations

While the model handles Levels 2–10 reliably, it still requires GPT fallback for deeper levels.

Generated maps are very good, but not yet high-quality or flawless — occasional imperfections remain.

Further scaling (larger datasets, additional fine-tuning passes) may allow the local model to support more levels independently.