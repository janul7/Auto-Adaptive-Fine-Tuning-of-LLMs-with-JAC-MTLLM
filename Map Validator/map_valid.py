# map_valid.py

# Validates maps by reachability only, plus a minimum enemy-count check:
#   - Walls & small_obstacles are blocking.
#   - Player and enemy cells are ALWAYS treated as free (overlaps allowed).
#   - Each enemy must be reachable from the player via 4-directional moves.
#   - If len(enemies) < level["num_enemies"], the map is INVALID.


import sys
import json
from collections import deque
from typing import Dict, Tuple, Set

def _in_bounds(x: int, y: int, w: int, h: int) -> bool:
    return 0 <= x < w and 0 <= y < h

def _collect_blocked(walls: list, small_obstacles: list) -> Set[Tuple[int, int]]:
    blocked: Set[Tuple[int, int]] = set()
    for wall in walls or []:
        x1, y1 = int(wall["start_pos"]["x"]), int(wall["start_pos"]["y"])
        x2, y2 = int(wall["end_pos"]["x"]), int(wall["end_pos"]["y"])
        if x1 == x2:  # vertical
            ys, ye = sorted((y1, y2))
            for y in range(ys, ye + 1):
                blocked.add((x1, y))
        elif y1 == y2:  # horizontal
            xs, xe = sorted((x1, x2))
            for x in range(xs, xe + 1):
                blocked.add((x, y1))
        else:
            # ignore diagonals if any
            pass
    for obs in small_obstacles or []:
        blocked.add((int(obs["x"]), int(obs["y"])))
    return blocked

def validate_map_json(data: Dict) -> bool:
    try:
        level = data["level"]
        width = int(level["width"])
        height = int(level["height"])
        required_enemies = int(level.get("num_enemies", 0))

        # Build blocking set from walls + small obstacles
        blocked = _collect_blocked(data.get("walls", []), data.get("small_obstacles", []))

        # Player start
        player = data["player_pos"]
        start = (int(player["x"]), int(player["y"]))
        if not _in_bounds(start[0], start[1], width, height):
            return False  # player must at least be on the board

        # Enemies list (also must be on the board)
        enemies = [(int(e["x"]), int(e["y"])) for e in data.get("enemies", [])]

        # --- NEW RULE: minimum enemy count must be met ---
        if len(enemies) < required_enemies:
            return False

        if any(not _in_bounds(ex, ey, width, height) for ex, ey in enemies):
            return False

        # OVERLAPS ALLOWED: treat player/enemy cells as free by removing them from blocked
        blocked.difference_update({start, *enemies})

        # BFS from player through free cells (4-neighborhood)
        q = deque([start])
        visited = {start}
        dirs = [(1,0), (-1,0), (0,1), (0,-1)]

        while q:
            x, y = q.popleft()
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if _in_bounds(nx, ny, width, height) and (nx, ny) not in blocked and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny))

        # Every enemy must be reachable
        return all((ex, ey) in visited for ex, ey in enemies)

    except Exception:
        return False

def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                outer = json.loads(line)
                # Each line is a record; the "output" field holds the actual map JSON string
                map_obj = json.loads(outer["output"])
                yield lineno, map_obj, None
            except Exception as e:
                yield lineno, None, e

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python map_valid.py <dataset.jsonl>")
        sys.exit(2)

    path = sys.argv[1]
    valid_count = invalid_count = parse_err = 0

    for lineno, map_obj, err in _iter_jsonl(path):
        if err is not None or map_obj is None:
            print(f"{lineno}\tINVALID\t(parse error)")
            parse_err += 1
            continue
        ok = validate_map_json(map_obj)
        print(f"{lineno}\t{'VALID' if ok else 'INVALID'}")
        if ok:
            valid_count += 1
        else:
            invalid_count += 1

    total = valid_count + invalid_count + parse_err
    print(f"Summary: {valid_count} valid, {invalid_count} invalid, {parse_err} parse errors; total {total}")
