
# LevelManager Map Fix — Preventing `list index out of range`

This README documents a small but critical fix to the **`LevelManager.get_map`** routine (Jac) that was causing a crash: **`list index out of range`**. It also shows how to reproduce the failure, what changed, and how to validate the fix.

> **TL;DR**
> - Root cause: writes to `map_tiles[y][x]` without bounds checks when walls/obstacles/enemies/player coordinates fall outside the `width × height` grid or when inclusive ranges step out of bounds.  
> - Fix: guard every write with `0 ≤ x < width` and `0 ≤ y < height`. Keep border padding as a *post* step.  
> - Result: no more crashes; invalid coordinates are ignored safely.

---

## 🔥 The Issue

The game crashed with `list index out of range` when building the tile map. This happens when a wall's end coordinate (inclusive loops) or any entity coordinate references a cell outside the grid.


**Exception location:**

```jac
map_tiles[y][x] = 'B';
```

### Why it happens

- **Inclusive ranges** (`end_pos.x + 1`, `end_pos.y + 1`) can step one past the valid index.  
- **LLM‑generated content** (or dynamic maps) may emit coordinates that sit on the right/bottom boundary (`x == width` or `y == height`) or even negative values.  
- **Borders are added later**, but writes happen *before* padding, so the raw grid must be respected.

---

## ✅ The Fix

Every write to `map_tiles` is now guarded by width/height checks:

```jac
if 0 <= x < map.level.width and 0 <= y < map.level.height {
    map_tiles[y][x] = 'B';
}
```

The same pattern is applied to **small obstacles**, **enemies**, and the **player**. Border padding remains as a final step so the core grid cannot be corrupted.

--- 
## fixed image 
<img width="1919" height="854" alt="How we fixed list index out of range error" src="https://github.com/user-attachments/assets/dd13cc67-c333-42e3-b889-4a537faa6fea" />

---


## 📁 Assets

- Screenshot: `assets/index_out_of_range.png` (included in this repo)
<img width="1211" height="354" alt="Screenshot 2025-08-23 165008" src="https://github.com/user-attachments/assets/71345979-e3d6-4e83-982a-f843a79afb17" />
