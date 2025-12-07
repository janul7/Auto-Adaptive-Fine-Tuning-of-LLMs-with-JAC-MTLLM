# 🗺️ Map Validator

`map_valid.py` is a Python utility that checks the **playability** of generated game maps.  
It enforces simple but critical rules to make sure maps are **valid**, **reachable**.

---

## ✅ Validation Rules

A map is considered **VALID** if all of the following hold:

1. **Player reachability**
   - The player can reach **every enemy** via 4-way movement (up, down, left, right).
   - A **Breadth-First Search (BFS)** is used to explore the grid.

2. **Blocking logic**
   - **Walls** (expanded across horizontal/vertical segments) and **small obstacles** block movement.
   - **Diagonal walls** are ignored (not supported).
   - Player and enemy cells are **always treated as free**, even if they overlap with obstacles.

3. **Enemy count**
   - The map must have **at least** `level["num_enemies"]` enemies.
   - Having more than the required enemies is allowed.

4. **Bounds**
   - Player and enemy positions must be **within the map dimensions** defined by `width` and `height`.

---

## ⚡ Usage

### Command line
```bash
python map_valid.py dataset.jsonl
```

# 📂 Example Input (JSONL)
```bash
{"output": "{\"level\":{\"name\":1,\"difficulty\":2,\"time\":300,\"width\":10,\"height\":10,\"num_wall\":5,\"num_enemies\":2},\"walls\":[{\"start_pos\":{\"x\":0,\"y\":0},\"end_pos\":{\"x\":0,\"y\":9}}],\"small_obstacles\":[{\"x\":3,\"y\":3}],\"enemies\":[{\"x\":5,\"y\":5},{\"x\":7,\"y\":8}],\"player_pos\":{\"x\":1,\"y\":1}}"}
```
# 📜 Example Output
```bash
1   VALID
2   INVALID
3   INVALID    (parse error)
Summary: 1 valid, 2 invalid, 1 parse errors; total 4
```

# 🛠️ How It Works
  - Parses JSON input → extracts level, walls, small_obstacles, enemies, and player_pos.
  - Builds blocked set → expands walls and marks obstacles as blocked cells.
  - Removes free cells → player and enemy positions are removed from the blocked set.
  - Runs BFS → explores the grid from the player position through free cells.
  - Checks enemies → all enemies must be found in the visited set.
  - Reports result → prints per-line result and a summary.

<img width="1483" height="232" alt="Screenshot 2025-09-03 022616" src="https://github.com/user-attachments/assets/167e69c9-833a-41ea-a309-8f5898869f9b" />




