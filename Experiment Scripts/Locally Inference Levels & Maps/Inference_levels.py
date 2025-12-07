import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = r"D:\Jac\TinyLlama-1.1B-Chat-v1.0"  # merged folder

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype="auto",   # float32 on CPU; mixed precision on GPU
    device_map="auto"
)

# Define 4 system/user pairs
prompts = [
    {
        "system": "This is a task you must complete by returning only the output.\nDo not include explanations, code, or extra text—only the result.\n",
        "user": "create_next_level\n\nself = LevelManager(current_level=2, current_difficulty=1, prev_levels=[], prev_level_maps=[])\nlast_levels = []\ndifficulty = 1"
    },
    {
        "system": "This is a task you must complete by returning only the output.\nDo not include explanations, code, or extra text—only the result.\n",
        "user": "create_next_level\n\nself = LevelManager(current_level=3, current_difficulty=2, prev_levels=[Level(name=3, difficulty=1, time=300, width=10, height=10, num_wall=5, num_enemies=2)], prev_level_maps=[Map_tiles(level=Level(name=3, difficulty=1, time=300, width=10, height=10, num_wall=5, num_enemies=2), walls=[Wall(start_pos=Position(x=0, y=0), end_pos=Position(x=0, y=9)), Wall(start_pos=Position(x=0, y=0), end_pos=Position(x=9, y=0)), Wall(start_pos=Position(x=9, y=0), end_pos=Position(x=9, y=9)), Wall(start_pos=Position(x=0, y=9), end_pos=Position(x=9, y=9)), Wall(start_pos=Position(x=5, y=4), end_pos=Position(x=5, y=5))], small_obstacles=[Position(x=7, y=1), Position(x=1, y=2), Position(x=8, y=3), Position(x=2, y=5), Position(x=8, y=4)], enemies=[Position(x=1, y=6), Position(x=5, y=2)], player_pos=Position(x=6, y=4))])\nlast_levels = [Level(name=3, difficulty=1, time=300, width=10, height=10, num_wall=5, num_enemies=2)]\ndifficulty = 2"
    },
    {
        "system": "This is a task you must complete by returning only the output.\nDo not include explanations, code, or extra text—only the result.\n",
        "user": "create_next_level\n\nself = LevelManager(current_level=4, current_difficulty=2, prev_levels=[Level(name=3, difficulty=1, time=300, width=10, height=10, num_wall=5, num_enemies=2), Level(name=4, difficulty=2, time=400, width=12, height=12, num_wall=6, num_enemies=3)], prev_level_maps=[Map_tiles(level=Level(name=3, difficulty=1, time=300, width=10, height=10, num_wall=5, num_enemies=2), walls=[Wall(start_pos=Position(x=0, y=0), end_pos=Position(x=0, y=9)), Wall(start_pos=Position(x=0, y=0), end_pos=Position(x=9, y=0)), Wall(start_pos=Position(x=9, y=0), end_pos=Position(x=9, y=9)), Wall(start_pos=Position(x=0, y=9), end_pos=Position(x=9, y=9)), Wall(start_pos=Position(x=5, y=4), end_pos=Position(x=5, y=5))], small_obstacles=[Position(x=7, y=1), Position(x=1, y=2), Position(x=8, y=3), Position(x=2, y=5), Position(x=8, y=4)], enemies=[Position(x=1, y=6), Position(x=5, y=2)], player_pos=Position(x=6, y=4)), Map_tiles(level=Level(name=4, difficulty=2, time=400, width=12, height=12, num_wall=6, num_enemies=3), walls=[Wall(start_pos=Position(x=0, y=0), end_pos=Position(x=0, y=11)), Wall(start_pos=Position(x=0, y=0), end_pos=Position(x=11, y=0)), Wall(start_pos=Position(x=11, y=0), end_pos=Position(x=11, y=11)), Wall(start_pos=Position(x=0, y=11), end_pos=Position(x=11, y=11)), Wall(start_pos=Position(x=6, y=5), end_pos=Position(x=6, y=6)), Wall(start_pos=Position(x=4, y=5), end_pos=Position(x=4, y=6))], small_obstacles=[Position(x=5, y=5), Position(x=4, y=9), Position(x=3, y=8), Position(x=4, y=1), Position(x=9, y=5)], enemies=[Position(x=10, y=2), Position(x=5, y=7), Position(x=3, y=5)], player_pos=Position(x=1, y=7))])\nlast_levels = [Level(name=3, difficulty=1, time=300, width=10, height=10, num_wall=5, num_enemies=2), Level(name=4, difficulty=2, time=400, width=12, height=12, num_wall=6, num_enemies=3)]\ndifficulty = 2"
    },
    {
        "system": "This is a task you must complete by returning only the output.\nDo not include explanations, code, or extra text—only the result.\n",
        "user": "create_next_level\n\nself = LevelManager(current_level=5, current_difficulty=3, prev_levels=[Level(name=3, difficulty=1, time=300, width=10, height=10, num_wall=5, num_enemies=2), Level(name=4, difficulty=2, time=400, width=12, height=12, num_wall=6, num_enemies=3), Level(name=5, difficulty=3, time=500, width=14, height=14, num_wall=7, num_enemies=4)], prev_level_maps=[Map_tiles(level=Level(name=3, difficulty=1, time=300, width=10, height=10, num_wall=5, num_enemies=2), walls=[Wall(start_pos=Position(x=0, y=0), end_pos=Position(x=0, y=9)), Wall(start_pos=Position(x=0, y=0), end_pos=Position(x=9, y=0)), Wall(start_pos=Position(x=9, y=0), end_pos=Position(x=9, y=9)), Wall(start_pos=Position(x=0, y=9), end_pos=Position(x=9, y=9)), Wall(start_pos=Position(x=5, y=4), end_pos=Position(x=5, y=5))], small_obstacles=[Position(x=7, y=1), Position(x=1, y=2), Position(x=8, y=3), Position(x=2, y=5), Position(x=8, y=4)], enemies=[Position(x=1, y=6), Position(x=5, y=2)], player_pos=Position(x=6, y=4)), Map_tiles(level=Level(name=4, difficulty=2, time=400, width=12, height=12, num_wall=6, num_enemies=3), walls=[Wall(start_pos=Position(x=0, y=0), end_pos=Position(x=0, y=11)), Wall(start_pos=Position(x=0, y=0), end_pos=Position(x=11, y=0)), Wall(start_pos=Position(x=11, y=0), end_pos=Position(x=11, y=11)), Wall(start_pos=Position(x=0, y=11), end_pos=Position(x=11, y=11)), Wall(start_pos=Position(x=6, y=5), end_pos=Position(x=6, y=6)), Wall(start_pos=Position(x=4, y=5), end_pos=Position(x=4, y=6))], small_obstacles=[Position(x=5, y=5), Position(x=4, y=9), Position(x=3, y=8), Position(x=4, y=1), Position(x=9, y=5)], enemies=[Position(x=10, y=2), Position(x=5, y=7), Position(x=3, y=5)], player_pos=Position(x=1, y=7)), Map_tiles(level=Level(name=5, difficulty=3, time=500, width=14, height=14, num_wall=7, num_enemies=4), walls=[Wall(start_pos=Position(x=0, y=0), end_pos=Position(x=0, y=13)), Wall(start_pos=Position(x=0, y=0), end_pos=Position(x=13, y=0)), Wall(start_pos=Position(x=13, y=0), end_pos=Position(x=13, y=13)), Wall(start_pos=Position(x=0, y=13), end_pos=Position(x=13, y=13)), Wall(start_pos=Position(x=7, y=6), end_pos=Position(x=7, y=7)), Wall(start_pos=Position(x=5, y=6), end_pos=Position(x=5, y=7)), Wall(start_pos=Position(x=8, y=7), end_pos=Position(x=10, y=7))], small_obstacles=[Position(x=6, y=4), Position(x=9, y=6), Position(x=3, y=3), Position(x=7, y=12), Position(x=8, y=4), Position(x=1, y=4), Position(x=11, y=8)], enemies=[Position(x=3, y=1), Position(x=5, y=4), Position(x=8, y=8), Position(x=7, y=10)], player_pos=Position(x=3, y=5))])\nlast_levels = [Level(name=3, difficulty=1, time=300, width=10, height=10, num_wall=5, num_enemies=2), Level(name=4, difficulty=2, time=400, width=12, height=12, num_wall=6, num_enemies=3), Level(name=5, difficulty=3, time=500, width=14, height=14, num_wall=7, num_enemies=4)]\ndifficulty = 3"
    }
]

# Loop through each prompt
for idx, p in enumerate(prompts, start=1):
    prompt_text = f"{p['system']}\n{p['user']}\n"
    inputs = tok(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tok.eos_token_id
        )

    generated_text = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"\n=== OUTPUT {idx} ===")
    print(generated_text.strip())
