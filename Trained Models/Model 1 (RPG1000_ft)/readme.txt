RPG Level Generator – Fine-Tuned Model (1000 Rows)

This model was fine-tuned on a dataset of 1000 rows to generate RPG-style game levels.

Performance

Generates early-level maps (Level 1 and Level 2) with average quality.

Occasionally extends to Level 3, but rarely beyond that.

Usually terminates after Level 2, limiting progression depth.

Observations

The structures are not much improved compared to smaller training sets — overall map design remains average.

Levels appear playable but simple, with minimal complexity.

Generation stability is better than extremely small dataset runs, but still prone to abrupt endings.

Limitations

Average structure quality, lacking creativity and detail.

Fails to reach deeper levels (4+).

Some outputs feel repetitive or incomplete.

Next Steps

Increase training dataset size (2000+ rows) to improve level diversity and depth.

Introduce regularization or better sampling techniques to avoid early termination.

Experiment with longer training schedules for more consistent progression.