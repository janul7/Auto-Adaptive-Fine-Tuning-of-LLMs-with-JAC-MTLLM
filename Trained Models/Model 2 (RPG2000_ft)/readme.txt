RPG Level Generator – Fine-Tuned Model (2000 Rows)

This is the second fine-tuned model trained on a dataset of 2000 rows for RPG game level generation.

Performance

The model successfully generates RPG level maps with patterns similar to levels 2, 3, 4, 5, and 6.

The output levels are playable and consistent, but the complexity does not significantly increase between levels.

Overall, map quality is average: they are functional but lack the variety and sophistication needed for richer gameplay.

Observations

Increasing the dataset size from 1000 → 2000 rows improved stability (more levels generated correctly before error).

However, level-to-level differentiation remains limited, and designs still feel repetitive.

Some structural errors or generation inconsistencies appear after several maps.

Next Steps

Experiment with larger and more diverse datasets to encourage richer structural patterns.

Apply regularization techniques (e.g., L2 regularization, KL-divergence to base model) to improve training stability.

Explore curriculum learning (progressively harder training data) to enforce increasing complexity between levels.

Fine-tune with LoRA adapters at different ranks to see if higher-rank adapters capture better structural details.