import json
import re
import random
import os
import time
import uuid
from collections import deque
from typing import Dict, Any, Optional, List, Tuple, Set

SYSTEM_PROMPT = (
    "This is a task you must complete by returning only the output.\n"
    "Do not include explanations, code, or extra text—only the result.\n"
)

# ---------- Stage 1: infer next level config (matches examples) ----------
def derive_level_config(current_level: int, difficulty: int) -> Dict[str, Any]:
    """
    From examples:
      - name = current_level + 1
      - time: 300 @ diff=1, 400 @ diff=2, 500 @ diff=3, 600 @ diff=4 -> 300 + 100*(diff-1)
      - width/height: 10 @ diff=1, 12 @ diff=2, 14 @ diff=3, 16 @ diff=4 -> 10 + 2*(diff-1)
      - num_wall: 5 @ diff=1, 6 @ diff=2, 7 @ diff=3, 8 @ diff=4 -> 5 + (diff-1)
      - num_enemies: 2 @ diff=1, 3 @ diff=2, 4 @ diff=3, 5 @ diff=4 -> 2 + (diff-1)
    """
    name = current_level + 1
    return {
        "name": name,
        "difficulty": difficulty,
        "time": 300 + 100 * (difficulty - 1),
        "width": 10 + 2 * (difficulty - 1),
        "height": 10 + 2 * (difficulty - 1),
        "num_wall": 5 + (difficulty - 1),
        "num_enemies": 2 + (difficulty - 1),
    }

# ---------- Walls (fixed/deterministic for a given level) ----------
def _boundary_walls(width: int, height: int) -> List[Dict[str, Dict[str, int]]]:
    """Four border walls around the grid."""
    w_max, h_max = width - 1, height - 1
    return [
        {"start_pos": {"x": 0, "y": 0}, "end_pos": {"x": 0, "y": h_max}},
        {"start_pos": {"x": 0, "y": 0}, "end_pos": {"x": w_max, "y": 0}},
        {"start_pos": {"x": w_max, "y": 0}, "end_pos": {"x": w_max, "y": h_max}},
        {"start_pos": {"x": 0, "y": h_max}, "end_pos": {"x": w_max, "y": h_max}},
    ]

def _deterministic_internal_walls(level: Dict[str, Any]) -> List[Dict[str, Dict[str, int]]]:
    """
    Internal wall segments are derived ONLY from level metadata so they do not change run-to-run.
    For diff=1 (num_wall=5), this yields a short center segment (2 tiles).
    For larger counts, place short segments around the center deterministically.
    """
    count = max(0, level["num_wall"] - 4)
    w, h = level["width"], level["height"]
    cx, cy = w // 2, h // 2

    sig = (level["name"] * 31 + level["difficulty"] * 97) & 0xFFFFFFFF
    walls: List[Dict[str, Dict[str, int]]] = []
    used: Set[Tuple[int, int, int, int]] = set()

    def add_seg(x0, y0, x1, y1):
        x0, x1 = max(0, min(w - 1, x0)), max(0, min(w - 1, x1))
        y0, y1 = max(0, min(h - 1, y0)), max(0, min(h - 1, y1))
        key = tuple(sorted([(x0, y0), (x1, y1)]))
        if key in used:
            return False
        used.add(key)
        walls.append({"start_pos": {"x": x0, "y": y0}, "end_pos": {"x": x1, "y": y1}})
        return True

    if count > 0:
        # First segment: vertical 2-length centered
        add_seg(cx, max(1, cy - 1), cx, min(h - 2, cy))

    # Additional segments based on signature (deterministic pattern)
    offsets = [(-2, 0, 0), (2, 0, 1), (0, -2, 1), (0, 2, 0), (-3, -1, 0), (3, 1, 1)]
    i = 1
    for dx, dy, horiz in offsets:
        if i >= count:
            break
        if horiz ^ (sig & 1):
            # horizontal len 3
            y = max(1, min(h - 2, cy + dy))
            x0 = max(1, min(w - 3, cx + dx - 1))
            add_seg(x0, y, x0 + 2, y)
        else:
            # vertical len 2
            x = max(1, min(w - 2, cx + dx))
            y0 = max(1, min(h - 3, cy + dy - 1))
            add_seg(x, y0, x, y0 + 1)
        i += 1

    return walls[:count]

def _collect_wall_cells(walls: List[Dict[str, Dict[str, int]]]) -> Set[Tuple[int, int]]:
    """Expand line segments into individual blocked cells."""
    cells: Set[Tuple[int, int]] = set()
    for w in walls:
        x0, y0 = w["start_pos"]["x"], w["start_pos"]["y"]
        x1, y1 = w["end_pos"]["x"], w["end_pos"]["y"]
        if x0 == x1:
            x = x0
            ys, ye = sorted([y0, y1])
            for y in range(ys, ye + 1):
                cells.add((x, y))
        elif y0 == y1:
            y = y0
            xs, xe = sorted([x0, x1])
            for x in range(xs, xe + 1):
                cells.add((x, y))
    return cells

# ---------- RNG (variety without exposing seed) ----------
def _rng_from_input(input_text: str) -> random.Random:
    """
    If a hidden 'seed = <int>' line exists in the input, use it (not printed back).
    Otherwise, create a strong, time-varying seed so maps differ run-to-run.
    """
    m = re.search(r"^\s*seed\s*=\s*(-?\d+)\s*$", input_text, re.IGNORECASE | re.MULTILINE)
    if m:
        return random.Random(int(m.group(1)))

    # Strong, varying seed from multiple entropy sources (never printed)
    try:
        seed_val = int.from_bytes(os.urandom(16), "big")
    except Exception:
        seed_val = 0
    seed_val ^= time.time_ns()
    seed_val ^= hash(uuid.uuid4())
    seed_val ^= (id(object()) << 16)
    return random.Random(seed_val & ((1 << 64) - 1))

# ---------- Helpers: geometry & pathfinding ----------
def _in_bounds(x: int, y: int, w: int, h: int) -> bool:
    return 0 <= x < w and 0 <= y < h

def _neighbors4(x: int, y: int) -> List[Tuple[int, int]]:
    return [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]

def _reachable_from(start: Tuple[int,int], w: int, h: int, blocked: Set[Tuple[int,int]]) -> Set[Tuple[int,int]]:
    if start in blocked or not _in_bounds(start[0], start[1], w, h):
        return set()
    seen: Set[Tuple[int,int]] = set([start])
    dq = deque([start])
    while dq:
        cx, cy = dq.popleft()
        for nx, ny in _neighbors4(cx, cy):
            if not _in_bounds(nx, ny, w, h): continue
            if (nx, ny) in blocked: continue
            if (nx, ny) in seen: continue
            seen.add((nx, ny))
            dq.append((nx, ny))
    return seen

def _chebyshev(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

# ---------- Placement helpers (vary ONLY positions) ----------
def _random_free_cell(rng: random.Random, width: int, height: int, forbidden: Set[Tuple[int, int]]) -> Tuple[int, int]:
    """Pick a random free cell (prefers interior, falls back to anywhere valid)."""
    interior = [(x, y) for y in range(1, height - 1) for x in range(1, width - 1) if (x, y) not in forbidden]
    if not interior:
        interior = [(x, y) for y in range(height) for x in range(width) if (x, y) not in forbidden]
    if not interior:
        return 0, 0
    return rng.choice(interior)

def _clustered_points(rng: random.Random, count: int, width: int, height: int,
                      forbidden: Set[Tuple[int,int]], cluster_count: int = 2) -> List[Dict[str,int]]:
    """Place points around a few cluster centers (without overlaps), with gentle jitter."""
    pts: List[Dict[str,int]] = []
    centers: List[Tuple[int,int]] = []
    for _ in range(cluster_count):
        cx, cy = rng.randint(1, max(1, width-2)), rng.randint(1, max(1, height-2))
        centers.append((cx, cy))
    attempts = 0
    while len(pts) < count and attempts < count*200:
        attempts += 1
        cx, cy = rng.choice(centers)
        dx = rng.randint(-2, 2)
        dy = rng.randint(-2, 2)
        x, y = cx + dx, cy + dy
        if not _in_bounds(x, y, width, height): continue
        if (x, y) in forbidden: continue
        forbidden.add((x, y))
        pts.append({"x": x, "y": y})
    if len(pts) < count:
        pts.extend(_uniform_points(rng, count - len(pts), width, height, forbidden))
    return pts

def _uniform_points(rng: random.Random, count: int, width: int, height: int,
                    forbidden: Set[Tuple[int,int]]) -> List[Dict[str,int]]:
    pts: List[Dict[str,int]] = []
    attempts = 0
    while len(pts) < count and attempts < count*200:
        attempts += 1
        x = rng.randint(0, width-1)
        y = rng.randint(0, height-1)
        if (x, y) in forbidden: continue
        forbidden.add((x, y))
        pts.append({"x": x, "y": y})
    if len(pts) < count:
        for yy in range(height):
            for xx in range(width):
                if (xx, yy) not in forbidden:
                    forbidden.add((xx, yy))
                    pts.append({"x": xx, "y": yy})
                    if len(pts) == count:
                        break
            if len(pts) == count: break
    return pts

def _place_small_obstacles(rng: random.Random, level: Dict[str,Any], walls_set: Set[Tuple[int,int]],
                           player_cell: Tuple[int,int]) -> List[Dict[str,int]]:
    """
    Place exactly num_wall small obstacles with style variation and connectivity guarantee.
    """
    w, h = level["width"], level["height"]
    count = level["num_wall"]
    cluster_prob = min(0.75, 0.3 + 0.15 * (level["difficulty"] - 1))
    max_tries = 40
    for _ in range(max_tries):
        forbidden = set(walls_set)
        forbidden.add(player_cell)

        style_clustered = rng.random() < cluster_prob
        if style_clustered and count >= 3:
            pts = _clustered_points(rng, count, w, h, forbidden, cluster_count=2)
        else:
            pts = _uniform_points(rng, count, w, h, forbidden)

        blocked = set(walls_set) | {(p["x"], p["y"]) for p in pts}
        reachable = _reachable_from(player_cell, w, h, blocked)
        free_cells = (w * h) - len(blocked)
        if free_cells == 0 or len(reachable) < max(1, int(0.3 * free_cells)):
            continue
        return pts
    forbidden = set(walls_set)
    forbidden.add(player_cell)
    return _uniform_points(rng, count, w, h, forbidden)

def _place_enemies(rng: random.Random, level: Dict[str,Any], walls_set: Set[Tuple[int,int]],
                   player_cell: Tuple[int,int], small_obstacles: List[Dict[str,int]]) -> List[Dict[str,int]]:
    """
    Place exactly num_enemies with spawn-safety and reachability guarantees.
    """
    w, h = level["width"], level["height"]
    count = level["num_enemies"]
    blocked = set(walls_set) | {(o["x"], o["y"]) for o in small_obstacles} | {player_cell}

    min_player_separation = 2  # Chebyshev distance >= 2
    max_tries = 200

    best: List[Dict[str,int]] = []
    for _ in range(max_tries):
        pts: List[Dict[str,int]] = []
        used = set(blocked)
        attempts = 0
        while len(pts) < count and attempts < count * 200:
            attempts += 1
            x = rng.randint(0, w-1)
            y = rng.randint(0, h-1)
            if (x, y) in used: continue
            if _chebyshev((x, y), player_cell) < min_player_separation: continue
            used.add((x, y))
            pts.append({"x": x, "y": y})

        if len(pts) < count:
            continue

        reach = _reachable_from(player_cell, w, h, set(walls_set) | {(o["x"], o["y"]) for o in small_obstacles})
        if any((e["x"], e["y"]) in reach for e in pts):
            return pts
        if not best:
            best = pts

    return best if best else []

# ---------- Stage 2: build map (vary ONLY positions) ----------
def build_map(level: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """
    - Walls are deterministic (do NOT change between runs for the same level).
    - Player, small_obstacles, enemies positions vary run-to-run (within bounds, no overlaps).
    - Quantities remain identical to the level spec.
    """
    width, height = level["width"], level["height"]
    walls = _boundary_walls(width, height) + _deterministic_internal_walls(level)
    wall_cells = _collect_wall_cells(walls)

    forbidden: Set[Tuple[int, int]] = set(wall_cells)

    # Player
    px, py = _random_free_cell(rng, width, height, forbidden)
    player_pos = {"x": px, "y": py}
    player_cell = (px, py)

    # Obstacles + Enemies
    small_obstacles = _place_small_obstacles(rng, level, wall_cells, player_cell)
    enemies_pts = _place_enemies(rng, level, wall_cells, player_cell, small_obstacles)

    return {
        "level": level,
        "walls": walls,
        "small_obstacles": small_obstacles,
        "enemies": enemies_pts,
        "player_pos": player_pos,
    }

# ---------- Parsing utilities (robust to nested brackets) ----------
CL_RE = re.compile(r"current_level\s*=\s*(\d+)", re.IGNORECASE)
CD_RE = re.compile(r"current_difficulty\s*=\s*(\d+)", re.IGNORECASE)

# NOTE: Use finditer and select the LAST 'level = Level(...)' occurrence.
# This ensures Stage-4 picks the explicit "level =" line, not the one inside prev_level_maps.
LEVEL_OBJ_RE = re.compile(
    r"level\s*=\s*Level\s*\(\s*name\s*=\s*(?P<name>\d+)\s*,\s*difficulty\s*=\s*(?P<diff>\d+)\s*,\s*time\s*=\s*(?P<time>\d+)\s*,\s*width\s*=\s*(?P<width>\d+)\s*,\s*height\s*=\s*(?P<height>\d+)\s*,\s*num_wall\s*=\s*(?P<num_wall>\d+)\s*,\s*num_enemies\s*=\s*(?P<num_enemies>\d+)\s*\)",
    re.IGNORECASE | re.DOTALL,
)

DIFF_LINE_RE = re.compile(r"^\s*difficulty\s*=\s*(?P<d>\d+)\s*$", re.IGNORECASE | re.MULTILINE)
MODE_RE = re.compile(r"^\s*(create_next_level|create_next_map)\s*$", re.IGNORECASE | re.MULTILINE)

# detect how many Level(...) are inside prev_levels=[ ... ]
PREV_LEVELS_RE = re.compile(r"prev_levels\s*=\s*\[(?P<body>.*?)\]", re.IGNORECASE | re.DOTALL)
def parse_prev_levels_count(input_block: str) -> int:
    m = PREV_LEVELS_RE.search(input_block)
    if not m:
        return 0
    body = m.group("body")
    return len(re.findall(r"\bLevel\s*\(", body, flags=re.IGNORECASE))

def parse_current_level(input_block: str) -> Optional[int]:
    m = CL_RE.search(input_block)
    return int(m.group(1)) if m else None

def parse_current_difficulty_from_manager(input_block: str) -> Optional[int]:
    m = CD_RE.search(input_block)
    return int(m.group(1)) if m else None

def parse_difficulty_explicit(input_block: str) -> Optional[int]:
    m = DIFF_LINE_RE.search(input_block)
    return int(m.group("d")) if m else None

def parse_level_object(input_block: str) -> Optional[Dict[str, Any]]:
    # Pick the LAST match to avoid grabbing Map_tiles(level=Level(...)) earlier in the text.
    last_m = None
    for m in LEVEL_OBJ_RE.finditer(input_block):
        last_m = m
    if not last_m:
        return None
    return {
        "name": int(last_m.group("name")),
        "difficulty": int(last_m.group("diff")),
        "time": int(last_m.group("time")),
        "width": int(last_m.group("width")),
        "height": int(last_m.group("height")),
        "num_wall": int(last_m.group("num_wall")),
        "num_enemies": int(last_m.group("num_enemies")),
    }

def parse_mode(input_block: str) -> Optional[str]:
    m = MODE_RE.search(input_block)
    return m.group(1) if m else None

# ---------- Sanitizer (do not echo any seed lines if present) ----------
def _sanitize_input_for_record(input_text: str) -> str:
    return re.sub(r"(?mi)^\s*seed\s*=\s*-?\d+\s*\n?", "", input_text).rstrip()

# ---------- Record builders (keep spaces in JSON like your samples) ----------
def build_record(system_text: str, input_text: str, output_obj: Dict[str, Any]) -> Dict[str, Any]:
    output_json_str = json.dumps(output_obj, separators=(", ", ": "), sort_keys=False)
    return {"system": system_text, "input": input_text, "output": output_json_str}

def generate_next_level_record(input_text: str) -> Dict[str, Any]:
    current_level = parse_current_level(input_text)
    difficulty = parse_difficulty_explicit(input_text) or parse_current_difficulty_from_manager(input_text)
    if current_level is None or difficulty is None:
        raise ValueError("Could not parse current_level and/or difficulty from input text.")

    # ----- Stage-5 & Stage-7 rule:
    # When prev_levels contains two levels (after Stage 4 map) OR three levels (after Stage 6 map),
    # bump the difficulty by +1 for the next Level config. This matches examples where Stage 5
    # produces difficulty 3 (from 2) and Stage 7 produces difficulty 4 (from 3).
    try:
        prev_count = parse_prev_levels_count(input_text)
        if prev_count >= 2:
            difficulty = difficulty + 1
    except Exception:
        pass
    # -------------------------------------------------------

    level = derive_level_config(current_level, difficulty)
    return build_record(SYSTEM_PROMPT, _sanitize_input_for_record(input_text), level)

def generate_next_map_record(input_text: str) -> Dict[str, Any]:
    rng = _rng_from_input(input_text)

    level_obj = parse_level_object(input_text)
    if level_obj is None:
        current_level = parse_current_level(input_text)
        difficulty = parse_difficulty_explicit(input_text) or parse_current_difficulty_from_manager(input_text)
        if current_level is None or difficulty is None:
            raise ValueError("Could not parse a Level(...) or LevelManager(...) from input text.")
        level_obj = derive_level_config(current_level, difficulty)

    map_obj = build_map(level_obj, rng)

    # ---- Stage-4 specific: force exactly 5 small_obstacles in output ----
    # Stage 4 pattern in your inputs: prev_levels contains EXACTLY two Level(...) entries.
    try:
        is_stage4 = parse_prev_levels_count(input_text) == 2
    except Exception:
        is_stage4 = False

    if is_stage4:
        target = 5  # force 5 small_obstacles in Stage 4 output
        cur = map_obj.get("small_obstacles", [])
        if len(cur) > target:
            map_obj["small_obstacles"] = cur[:target]
        elif len(cur) < target:
            need = target - len(cur)
            walls_set = _collect_wall_cells(map_obj["walls"])
            forbidden = set(walls_set)
            forbidden.add((map_obj["player_pos"]["x"], map_obj["player_pos"]["y"]))
            for p in cur:
                forbidden.add((p["x"], p["y"]))
            for e in map_obj["enemies"]:
                forbidden.add((e["x"], e["y"]))
            fillers = _uniform_points(rng, need, level_obj["width"], level_obj["height"], forbidden)
            map_obj["small_obstacles"].extend(fillers)
    # --------------------------------------------------------------------

    return build_record(SYSTEM_PROMPT, _sanitize_input_for_record(input_text), map_obj)

# ---------- Dispatcher ----------
def generate_record(input_text: str) -> Dict[str, Any]:
    mode = parse_mode(input_text)
    if mode is None:
        raise ValueError("Input must include either 'create_next_level' or 'create_next_map'.")
    mode = mode.lower()
    if mode == "create_next_level":
        return generate_next_level_record(input_text)
    elif mode == "create_next_map":
        return generate_next_map_record(input_text)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

# ---------- Formatting helpers to build Stage 3, 4, 5, 6 & 7 inputs from prior outputs ----------
def _fmt_level_for_input(level: Dict[str, Any]) -> str:
    return (
        f"Level(name={level['name']}, difficulty={level['difficulty']}, time={level['time']}, "
        f"width={level['width']}, height={level['height']}, num_wall={level['num_wall']}, "
        f"num_enemies={level['num_enemies']})"
    )

def _fmt_pos(x: int, y: int) -> str:
    return f"Position(x={x}, y={y})"

def _fmt_wall(w: Dict[str, Dict[str, int]]) -> str:
    sx, sy = w["start_pos"]["x"], w["start_pos"]["y"]
    ex, ey = w["end_pos"]["x"], w["end_pos"]["y"]
    return f"Wall(start_pos={_fmt_pos(sx, sy)}, end_pos={_fmt_pos(ex, ey)})"

def _fmt_map_tiles(map_obj: Dict[str, Any]) -> str:
    lvl_txt = _fmt_level_for_input(map_obj["level"])
    walls_txt = ", ".join(_fmt_wall(w) for w in map_obj["walls"])
    small_obs_txt = ", ".join(_fmt_pos(p["x"], p["y"]) for p in map_obj["small_obstacles"])
    enemies_txt = ", ".join(_fmt_pos(p["x"], p["y"]) for p in map_obj["enemies"])
    player_txt = _fmt_pos(map_obj["player_pos"]["x"], map_obj["player_pos"]["y"])
    return (
        f"Map_tiles(level={lvl_txt}, walls=[{walls_txt}], small_obstacles=[{small_obs_txt}], "
        f"enemies=[{enemies_txt}], player_pos={player_txt})"
    )

def build_stage3_input_from_map(next_difficulty: int,
                                prev_level: Dict[str, Any],
                                map_obj: Dict[str, Any]) -> str:
    """
    Build Stage-3 'create_next_level' input using EXACT Stage-2 positions.
    """
    lvl_txt = _fmt_level_for_input(prev_level)
    prev_level_maps_txt = _fmt_map_tiles(map_obj)

    input_str = (
        "create_next_level\n\n"
        f"self = LevelManager(current_level={prev_level['name']}, current_difficulty={next_difficulty}, "
        f"prev_levels=[{lvl_txt}], prev_level_maps=[{prev_level_maps_txt}])\n"
        f"last_levels = [{lvl_txt}]\n"
        f"difficulty = {next_difficulty}"
    )
    return input_str

def build_stage4_input_from_stage3(prev_level_for_map: Dict[str, Any],
                                   stage3_level_cfg: Dict[str, Any],
                                   stage2_map_obj: Dict[str, Any]) -> str:
    """
    Build Stage-4 'create_next_map' input.
    IMPORTANT: Use Stage-3 OUTPUT as the 'level =' argument (exact config).
    prev_levels includes [level3, level4], and prev_level_maps includes the Stage-2 map for level3.
    """
    lvl3_txt = _fmt_level_for_input(prev_level_for_map)      # Level 3
    lvl4_txt = _fmt_level_for_input(stage3_level_cfg)        # Level 4 (from Stage 3 OUTPUT)
    prev_level_maps_txt = _fmt_map_tiles(stage2_map_obj)

    input_str = (
        "create_next_map\n\n"
        f"self = LevelManager(current_level={prev_level_for_map['name']}, current_difficulty={stage3_level_cfg['difficulty']}, "
        f"prev_levels=[{lvl3_txt}, {lvl4_txt}], prev_level_maps=[{prev_level_maps_txt}])\n"
        f"level = {lvl4_txt}"
    )
    return input_str

def build_stage5_input_from_stage4(level3_map: Dict[str, Any],
                                   level4_map: Dict[str, Any]) -> str:
    """
    Build Stage-5 'create_next_level' input.
    - prev_levels includes [Level 3, Level 4]
    - prev_level_maps includes both maps: Map(Level 3), Map(Level 4)
    - current_level is Level 4's name
    - current_difficulty/difficulty lines reflect Level 4's difficulty (2)
      The Stage-5 rule in generate_next_level_record bumps it to 3.
    """
    lvl3 = level3_map["level"]
    lvl4 = level4_map["level"]
    lvl3_txt = _fmt_level_for_input(lvl3)
    lvl4_txt = _fmt_level_for_input(lvl4)
    map3_txt = _fmt_map_tiles(level3_map)
    map4_txt = _fmt_map_tiles(level4_map)

    input_str = (
        "create_next_level\n\n"
        f"self = LevelManager(current_level={lvl4['name']}, current_difficulty={lvl4['difficulty']}, "
        f"prev_levels=[{lvl3_txt}, {lvl4_txt}], prev_level_maps=[{map3_txt}, {map4_txt}])\n"
        f"last_levels = [{lvl3_txt}, {lvl4_txt}]\n"
        f"difficulty = {lvl4['difficulty']}"
    )
    return input_str

def build_stage6_input_from_stage5(level3_map: Dict[str, Any],
                                   level4_map: Dict[str, Any],
                                   stage5_level_cfg: Dict[str, Any]) -> str:
    """
    Build Stage-6 'create_next_map' input (for Level 5).
    - current_level is Level 4's name
    - current_difficulty is Level 4's difficulty
    - prev_levels includes [Level 3, Level 4, Level 5]
    - prev_level_maps includes Map(Level 3) and Map(Level 4)
    - level = Level 5 (the Stage-5 OUTPUT config)
    """
    lvl3 = level3_map["level"]
    lvl4 = level4_map["level"]
    lvl5 = stage5_level_cfg

    lvl3_txt = _fmt_level_for_input(lvl3)
    lvl4_txt = _fmt_level_for_input(lvl4)
    lvl5_txt = _fmt_level_for_input(lvl5)

    map3_txt = _fmt_map_tiles(level3_map)
    map4_txt = _fmt_map_tiles(level4_map)

    input_str = (
        "create_next_map\n\n"
        f"self = LevelManager(current_level={lvl4['name']}, current_difficulty={lvl4['difficulty']}, "
        f"prev_levels=[{lvl3_txt}, {lvl4_txt}, {lvl5_txt}], prev_level_maps=[{map3_txt}, {map4_txt}])\n"
        f"level = {lvl5_txt}"
    )
    return input_str

def build_stage7_input_from_stage6(level3_map: Dict[str, Any],
                                   level4_map: Dict[str, Any],
                                   level5_map: Dict[str, Any]) -> str:
    """
    Build Stage-7 'create_next_level' input (for Level 6).
    - prev_levels includes [Level 3, Level 4, Level 5]
    - prev_level_maps includes [Map(Level 3), Map(Level 4), Map(Level 5)]
    - current_level is Level 5's name
    - current_difficulty/difficulty reflect Level 5's difficulty (3)
      The Stage-7 rule in generate_next_level_record bumps it to 4.
    """
    lvl3 = level3_map["level"]
    lvl4 = level4_map["level"]
    lvl5 = level5_map["level"]

    lvl3_txt = _fmt_level_for_input(lvl3)
    lvl4_txt = _fmt_level_for_input(lvl4)
    lvl5_txt = _fmt_level_for_input(lvl5)

    map3_txt = _fmt_map_tiles(level3_map)
    map4_txt = _fmt_map_tiles(level4_map)
    map5_txt = _fmt_map_tiles(level5_map)

    input_str = (
        "create_next_level\n\n"
        f"self = LevelManager(current_level={lvl5['name']}, current_difficulty={lvl5['difficulty']}, "
        f"prev_levels=[{lvl3_txt}, {lvl4_txt}, {lvl5_txt}], prev_level_maps=[{map3_txt}, {map4_txt}, {map5_txt}])\n"
        f"last_levels = [{lvl3_txt}, {lvl4_txt}, {lvl5_txt}]\n"
        f"difficulty = {lvl5['difficulty']}"
    )
    return input_str

# ---------- NEW: Stage 8 input builder (map for Level 6) ----------
def build_stage8_input_from_stage7(level3_map: Dict[str, Any],
                                   level4_map: Dict[str, Any],
                                   level5_map: Dict[str, Any],
                                   stage7_level_cfg: Dict[str, Any]) -> str:
    """
    Build Stage-8 'create_next_map' input (for Level 6), matching the examples' pattern.
    - current_level is Level 5's name
    - current_difficulty is Level 5's difficulty
    - prev_levels includes [Level 3, Level 4, Level 5, Level 6]
      where Level 6 is the Stage-7 OUTPUT config.
    - prev_level_maps includes [Map(Level 3), Map(Level 4), Map(Level 5)]
    - level = Level 6 (the Stage-7 OUTPUT config)
    """
    lvl3 = level3_map["level"]
    lvl4 = level4_map["level"]
    lvl5 = level5_map["level"]
    lvl6 = stage7_level_cfg  # from Stage 7 output

    lvl3_txt = _fmt_level_for_input(lvl3)
    lvl4_txt = _fmt_level_for_input(lvl4)
    lvl5_txt = _fmt_level_for_input(lvl5)
    lvl6_txt = _fmt_level_for_input(lvl6)

    map3_txt = _fmt_map_tiles(level3_map)
    map4_txt = _fmt_map_tiles(level4_map)
    map5_txt = _fmt_map_tiles(level5_map)

    input_str = (
        "create_next_map\n\n"
        f"self = LevelManager(current_level={lvl5['name']}, current_difficulty={lvl5['difficulty']}, "
        f"prev_levels=[{lvl3_txt}, {lvl4_txt}, {lvl5_txt}, {lvl6_txt}], prev_level_maps=[{map3_txt}, {map4_txt}, {map5_txt}])\n"
        f"level = {lvl6_txt}"
    )
    return input_str

# ---------- Example usage: generate Stage 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 chain ----------
if __name__ == "__main__":
    # Stage 1 (create_next_level)
    stage1_input = (
        "create_next_level\n\n"
        "self = LevelManager(current_level=2, current_difficulty=1, prev_levels=[], prev_level_maps=[])\n"
        "last_levels = []\n"
        "difficulty = 1"
    )
    rec1 = generate_record(stage1_input)
    print(json.dumps(rec1, ensure_ascii=False))

    # Stage 2 (create_next_map)
    stage2_input = (
        "create_next_map\n\n"
        "self = LevelManager(current_level=2, current_difficulty=1, prev_levels=[Level(name=3, difficulty=1, time=300, width=10, height=10, num_wall=5, num_enemies=2)], prev_level_maps=[])\n"
        "level = Level(name=3, difficulty=1, time=300, width=10, height=10, num_wall=5, num_enemies=2)"
    )
    rec2 = generate_record(stage2_input)
    print(json.dumps(rec2, ensure_ascii=False))

    # Stage 3 (create_next_level)
    stage2_out = json.loads(rec2["output"])      # dict with level, walls, small_obstacles, enemies, player_pos
    prev_level = stage2_out["level"]             # Level 3 dict
    next_difficulty = 2                          # as in examples
    stage3_input = build_stage3_input_from_map(next_difficulty, prev_level, stage2_out)
    rec3 = generate_record(stage3_input)
    print(json.dumps(rec3, ensure_ascii=False))

    # Stage 4 (create_next_map)
    stage3_level_cfg = json.loads(rec3["output"])  # Level 4 config dict FROM STAGE 3
    stage4_input = build_stage4_input_from_stage3(
        prev_level_for_map=prev_level,             # Level 3
        stage3_level_cfg=stage3_level_cfg,         # Level 4 (from Stage 3 output)
        stage2_map_obj=stage2_out                  # Map of Level 3
    )
    rec4 = generate_record(stage4_input)
    print(json.dumps(rec4, ensure_ascii=False))

    # Stage 5 (create_next_level)
    stage4_out = json.loads(rec4["output"])       # dict with level 4 map
    stage5_input = build_stage5_input_from_stage4(stage2_out, stage4_out)
    rec5 = generate_record(stage5_input)
    print(json.dumps(rec5, ensure_ascii=False))

    # Stage 6 (create_next_map) for Level 5
    stage5_level_cfg = json.loads(rec5["output"])  # Level 5 config dict FROM STAGE 5
    stage6_input = build_stage6_input_from_stage5(stage2_out, stage4_out, stage5_level_cfg)
    rec6 = generate_record(stage6_input)
    print(json.dumps(rec6, ensure_ascii=False))

    # Stage 7 (create_next_level) for Level 6
    stage6_out = json.loads(rec6["output"])        # Map for Level 5
    stage7_input = build_stage7_input_from_stage6(stage2_out, stage4_out, stage6_out)
    rec7 = generate_record(stage7_input)
    print(json.dumps(rec7, ensure_ascii=False))

    # Stage 8 (create_next_map) for Level 6
    stage7_level_cfg = json.loads(rec7["output"])  # Level 6 config dict FROM STAGE 7
    stage8_input = build_stage8_input_from_stage7(stage2_out, stage4_out, stage6_out, stage7_level_cfg)
    rec8 = generate_record(stage8_input)
    print(json.dumps(rec8, ensure_ascii=False))
