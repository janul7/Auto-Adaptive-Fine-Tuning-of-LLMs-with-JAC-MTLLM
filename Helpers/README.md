# Helpers

Utility scripts for preparing datasets and models in the RPG Game Generation fine-tuning project.

## Scripts
- **category_extractor.py**  
  Groups a JSONL dataset into 18 categories with row limits for each. Used to prepare balanced data for fine-tuning.

- **merge_peft.py**  
  Merges a fine-tuned PEFT/LoRA adapter into the base model and saves a standalone merged model.

- **token_count_checker.py**  
  Checks token counts of `system + input + output` samples to ensure they fit within model context limits.

