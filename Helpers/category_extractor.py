import json

input_file = "RPG_Game_DataSet3.jsonl"
output_file = "DataSet_Grouped(60,000_Rows).jsonl" 

# Define how many rows to collect per category (1–18)
limits = {
    1: 1000,  2: 1000,
    3: 1500,  4: 1500, 
    5: 2200,  6: 2200, 
    7: 2700,  8: 2700, 
    9: 3300, 10: 3300, 
    11: 3800, 12: 3800, 
    13: 4300, 14: 4300, 
    15: 5000, 16: 5000, 
    17: 6200, 18: 6200 
}

selected_lines = []
counts = {k: 0 for k in limits}

# Read the whole file once into memory (so we can scan it multiple times)
with open(input_file, "r", encoding="utf-8") as f:
    all_lines = f.readlines()

# Now collect category by category
for cat in range(1, 19):  # 1..18
    needed = limits[cat]
    for idx, line in enumerate(all_lines, start=1):
        category = ((idx - 1) % 18) + 1
        if category == cat and counts[cat] < needed:
            selected_lines.append(line)
            counts[cat] += 1
        if counts[cat] >= needed:
            break  # done with this category, move to the next

# Write output file
with open(output_file, "w", encoding="utf-8") as f:
    for line in selected_lines:
        f.write(line)

print(f"Saved {len(selected_lines)} rows to {output_file}")
print("Distribution:", counts)
