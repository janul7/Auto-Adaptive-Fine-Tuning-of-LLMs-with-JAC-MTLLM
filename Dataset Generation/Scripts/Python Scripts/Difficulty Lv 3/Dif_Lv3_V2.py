# new_newwalls_lv11_final.py  (connectivity- and placement-tuned)
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

# ---------- Stage 1: infer next level config ----------
def derive_level_config(current_level: int, difficulty: int) -> Dict[str, Any]:
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

# ---------- Helpers ----------
def _in_bounds(x: int, y: int, w: int, h: int) -> bool:
    return 0 <= x < w and 0 <= y < h

def _neighbors4(x: int, y: int) -> List[Tuple[int, int]]:
    return [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]

def _segment_cells_len(seg: Dict[str, Dict[str, int]]) -> int:
    x0, y0 = seg["start_pos"]["x"], seg["start_pos"]["y"]
    x1, y1 = seg["end_pos"]["x"], seg["end_pos"]["y"]
    return abs(x1 - x0) + abs(y1 - y0) + 1

def _segment_len_bounds(width: int, height: int) -> Tuple[int, int]:
    """
    For square maps, enforce: (N/2)-2 < length < (N/2)+2, where length is number of cells.
    That is: allowed inclusive range is [N/2-1, N/2+1].
    Examples:
      16x16 -> 7..9
      18x18 -> 8..10
      20x20 -> 9..11
      22x22 -> 10..12
      24x24 -> 11..13
      26x26 -> 12..14
    """
    n = min(width, height)
    return (n // 2 - 1, n // 2 + 1)

def _collect_wall_cells(walls: List[Dict[str, Dict[str, int]]]) -> Set[Tuple[int, int]]:
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

def _all_free_components(w: int, h: int, blocked: Set[Tuple[int,int]]) -> List[Set[Tuple[int,int]]]:
    """Connected components of free space (blocked = walls only)."""
    seen: Set[Tuple[int,int]] = set()
    comps: List[Set[Tuple[int,int]]] = []
    for y in range(h):
        for x in range(w):
            if (x, y) in blocked or (x, y) in seen:
                continue
            comp = _reachable_from((x, y), w, h, blocked)
            if comp:
                comps.append(comp)
                seen |= comp
    return comps

def _largest_component(w: int, h: int, blocked: Set[Tuple[int,int]]) -> Set[Tuple[int,int]]:
    comps = _all_free_components(w, h, blocked)
    return max(comps, key=len) if comps else set()

# ---------- RNG ----------
def _rng_from_input(input_text: str) -> random.Random:
    m = re.search(r"^\s*seed\s*=\s*(-?\d+)\s*$", input_text, re.IGNORECASE | re.MULTILINE)
    if m:
        return random.Random(int(m.group(1)))
    try:
        seed_val = int.from_bytes(os.urandom(16), "big")
    except Exception:
        seed_val = 0
    seed_val ^= time.time_ns()
    seed_val ^= hash(uuid.uuid4())
    seed_val ^= (id(object()) << 16)
    return random.Random(seed_val & ((1 << 64) - 1))

# ---------- Placement ----------
def _random_free_cell(rng: random.Random, width: int, height: int, forbidden: Set[Tuple[int, int]]) -> Tuple[int, int]:
    interior = [(x, y) for y in range(1, height - 1) for x in range(1, width - 1) if (x, y) not in forbidden]
    if not interior:
        interior = [(x, y) for y in range(height) for x in range(width) if (x, y) not in forbidden]
    if not interior:
        return 0, 0
    return rng.choice(interior)

def _uniform_points(rng: random.Random, count: int, width: int, height: int,
                    forbidden: Set[Tuple[int,int]]) -> List[Dict[str,int]]:
    pts: List[Dict[str,int]] = []
    attempts = 0
    while len(pts) < count and attempts < count*200:
        attempts += 1
        x = rng.randint(0, width-1); y = rng.randint(0, height-1)
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

def _clustered_points(rng: random.Random, count: int, width: int, height: int,
                      forbidden: Set[Tuple[int,int]], cluster_count: int = 2) -> List[Dict[str,int]]:
    pts: List[Dict[str,int]] = []
    centers: List[Tuple[int,int]] = []
    for _ in range(cluster_count):
        cx, cy = rng.randint(1, max(1, width-2)), rng.randint(1, max(1, height-2))
        centers.append((cx, cy))
    attempts = 0
    while len(pts) < count and attempts < count*200:
        attempts += 1
        cx, cy = rng.choice(centers)
        dx = rng.randint(-2, 2); dy = rng.randint(-2, 2)
        x, y = cx + dx, cy + dy
        if not _in_bounds(x, y, width, height): continue
        if (x, y) in forbidden: continue
        forbidden.add((x, y))
        pts.append({"x": x, "y": y})
    if len(pts) < count:
        pts.extend(_uniform_points(rng, count - len(pts), width, height, forbidden))
    return pts

def _place_small_obstacles(rng: random.Random, level: Dict[str,Any], walls_set: Set[Tuple[int,int]],
                           player_cell: Tuple[int,int]) -> List[Dict[str,int]]:
    """
    Small obstacles may not overlap walls, player, or enemies.
    Enemies are placed after obstacles, so we block enemy overlap in enemy placement.
    """
    w, h = level["width"], level["height"]
    count = level["num_wall"]  # follows your dataset pattern
    cluster_prob = min(0.75, 0.3 + 0.15 * (level["difficulty"] - 1))
    max_tries = 60
    # Require the player's reachable area (with walls+obstacles) to be a good share of open cells.
    # This prevents maps where the player starts in a tiny pocket.
    # (Keeps existing behavior but filters out cramped layouts.)
    target_frac = min(0.7, 0.4 + 0.03 * (level["difficulty"]))  # ~0.4..0.7

    for _ in range(max_tries):
        forbidden = set(walls_set); forbidden.add(player_cell)
        style_clustered = rng.random() < cluster_prob
        if style_clustered and count >= 3:
            pts = _clustered_points(rng, count, w, h, forbidden, cluster_count=2)
        else:
            pts = _uniform_points(rng, count, w, h, forbidden)
        blocked = set(walls_set) | {(p["x"], p["y"]) for p in pts} | {player_cell}
        reachable = _reachable_from(player_cell, w, h, blocked)
        free_cells = (w * h) - len(blocked)
        if free_cells == 0 or len(reachable) < max(1, int(target_frac * free_cells)):
            continue
        return pts
    # Fallback if we couldn't hit the target spread:
    forbidden = set(walls_set); forbidden.add(player_cell)
    return _uniform_points(rng, count, w, h, forbidden)

def _place_enemies(rng: random.Random, level: Dict[str,Any], walls_set: Set[Tuple[int,int]],
                   player_cell: Tuple[int,int], small_obstacles: List[Dict[str,int]]) -> List[Dict[str,int]]:
    """
    Place enemies so they are well spread across the map (no clustering) and
    the player can reach the *exact* cell each enemy occupies (maze-like check).
    Constraints:
      - No overlap with walls, small obstacles, or player.
      - Pairwise Chebyshev separation between enemies is enforced and adapted to map size & count.
      - Also keep a minimum separation from the player.
      - Softly avoid corners/outer rim so enemies aren't always hugging borders.
    """
    w, h = level["width"], level["height"]
    n = level["num_enemies"]

    # Static blocks (enemies aren't considered blocking for reachability)
    static_blocked = set(walls_set) | {(o["x"], o["y"]) for o in small_obstacles}

    # Cells reachable by the player when considering walls + small obstacles only.
    reach = _reachable_from(player_cell, w, h, static_blocked)
    if not reach:
        return []

    # Candidate cells = all reachable cells except the player's.
    candidates = [c for c in reach if c != player_cell]
    if not candidates:
        return []

    import math
    base_sep = int(min(w, h) / (math.sqrt(max(1, n)) + 1)) - 1
    target_sep = max(3, min(base_sep, 7))
    min_player_sep = max(3, target_sep - 1)

    def cheb(a: Tuple[int,int], b: Tuple[int,int]) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def border_dist(c: Tuple[int,int]) -> int:
        x, y = c
        return min(x, w-1-x, y, h-1-y)

    # In most cases, avoid putting enemies on the outer 1-2 tiles; sometimes allow it.
    interior_margin = 2 if rng.random() < 0.7 else 0

    def farthest_sampling(min_sep: int) -> List[Tuple[int,int]]:
        """Greedy farthest-point sampling over 'candidates' with min_sep and player clearance."""
        chosen: List[Tuple[int,int]] = []

        pool0 = [c for c in candidates if cheb(c, player_cell) >= min_player_sep and border_dist(c) >= interior_margin]
        if not pool0:
            pool0 = [c for c in candidates if cheb(c, player_cell) >= min_player_sep]
            if not pool0:
                return []

        # Pick the first point among the farthest-from-player set (top 20%).
        dists = [(c, cheb(c, player_cell)) for c in pool0]
        dists.sort(key=lambda t: t[1], reverse=True)
        topk = max(1, len(dists) // 5)
        first = rng.choice([c for (c, _) in dists[:topk]])
        chosen.append(first)

        # Iteratively pick the candidate that maximizes min distance to existing chosen
        # while respecting min_sep (to enemies) and min_player_sep (to player).
        while len(chosen) < n:
            best_c = None
            best_score = -1

            pool = pool0
            if len(pool0) > 4000:
                pool = rng.sample(pool0, 4000)

            for c in pool:
                if cheb(c, player_cell) < min_player_sep:
                    continue
                if any(cheb(c, e) < min_sep for e in chosen):
                    continue
                # Score: spread from others (dominant), then prefer interior a bit
                score = min(cheb(c, e) for e in chosen) * 10 + border_dist(c)
                if score > best_score:
                    best_score = score
                    best_c = c

            if best_c is None:
                break
            chosen.append(best_c)

        return chosen

    # Try with decreasing separation until we can place all enemies.
    sep = target_sep
    result: List[Tuple[int,int]] = []
    while sep >= 2:
        picked = farthest_sampling(sep)
        if len(picked) >= n:
            result = picked[:n]
            break
        if len(picked) > len(result):
            result = picked[:]
        sep -= 1

    # If still not enough (very constrained maps), fill remaining using a gentler rule.
    if len(result) < n:
        remaining = n - len(result)
        placed = set(result)
        soft_sep = 2
        pool = [c for c in candidates if c not in placed and cheb(c, player_cell) >= 2 and border_dist(c) >= max(0, interior_margin-1)]
        rng.shuffle(pool)
        for c in pool:
            if len(result) >= n:
                break
            if all(cheb(c, e) >= soft_sep for e in placed):
                result.append(c)
                placed.add(c)

    return [{"x": x, "y": y} for (x, y) in result]

# ---------- Walls (length-constrained) ----------
def _random_segment_within_range(rng: random.Random, w: int, h: int,
                                 used: Set[Tuple[Tuple[int,int], Tuple[int,int]]],
                                 min_len: int, max_len: int) -> Optional[Dict[str, Dict[str, int]]]:
    """
    Generate a vertical or horizontal segment with number-of-cells length L,
    where min_len <= L <= max_len (inclusive). Avoid duplicates in 'used'.
    """
    for _ in range(300):
        vertical = rng.random() < 0.5
        L = rng.randint(min_len, max_len)  # cells count
        if vertical:
            x = rng.randint(0, w - 1)
            y0 = rng.randint(0, h - L)
            y1 = y0 + (L - 1)
            x0, y0n, x1, y1n = x, y0, x, y1
        else:
            y = rng.randint(0, h - 1)
            x0 = rng.randint(0, w - L)
            x1 = x0 + (L - 1)
            x0, x1 = min(x0, x1), max(x0, x1)
            y0n, y1n = y, y
        a, b = (x0, y0n), (x1, y1n)
        key = (a, b) if a <= b else (b, a)
        if key in used:
            continue
        used.add(key)
        return {"start_pos": {"x": x0, "y": y0n}, "end_pos": {"x": x1, "y": y1n}}
    return None

def _refit_segment_to_range(rng: random.Random, seg: Dict[str, Dict[str,int]],
                            w: int, h: int, min_len: int, max_len: int,
                            used: Set[Tuple[Tuple[int,int], Tuple[int,int]]]) -> Optional[Dict[str, Dict[str,int]]]:
    """
    Take an existing segment and produce a new one aligned to the same orientation
    that fits the length bounds. If it fails, generate a fresh segment.
    """
    x0, y0 = seg["start_pos"]["x"], seg["start_pos"]["y"]
    x1, y1 = seg["end_pos"]["x"], seg["end_pos"]["y"]
    vertical = (x0 == x1)
    for _ in range(200):
        L = rng.randint(min_len, max_len)
        if vertical:
            x = x0 if 0 <= x0 < w else rng.randint(0, w - 1)
            y_start = rng.randint(0, h - L)
            y_end = y_start + (L - 1)
            a, b = (x, y_start), (x, y_end)
        else:
            y = y0 if 0 <= y0 < h else rng.randint(0, h - 1)
            x_start = rng.randint(0, w - L)
            x_end = x_start + (L - 1)
            a, b = (x_start, y), (x_end, y)
        key = (a, b) if a <= b else (b, a)
        if key in used:
            continue
        used.add(key)
        return {"start_pos": {"x": a[0], "y": a[1]}, "end_pos": {"x": b[0], "y": b[1]}}
    return _random_segment_within_range(rng, w, h, used, min_len, max_len)

def _deterministic_internal_walls(level: Dict[str, Any]) -> List[Dict[str, Dict[str, int]]]:
    """
    Produce exactly level['num_wall'] internal segments whose lengths satisfy the size-based bounds.
    """
    count = level["num_wall"]
    w, h = level["width"], level["height"]
    min_len, max_len = _segment_len_bounds(w, h)

    # deterministic RNG from level signature for reproducibility
    sig = (level["name"] * 31 + level["difficulty"] * 97 + w * 7 + h * 13) & 0xFFFFFFFF
    rng = random.Random(sig)

    walls: List[Dict[str, Dict[str, int]]] = []
    used: Set[Tuple[Tuple[int,int], Tuple[int,int]]] = set()

    for _ in range(count):
        seg = _random_segment_within_range(rng, w, h, used, min_len, max_len)
        if seg is None:
            break
        walls.append(seg)

    # Final safety: enforce lengths strictly within bounds
    ok_walls: List[Dict[str, Dict[str, int]]] = []
    used2: Set[Tuple[Tuple[int,int], Tuple[int,int]]] = set()
    for seg in walls:
        L = _segment_cells_len(seg)
        if not (min_len <= L <= max_len):
            seg = _refit_segment_to_range(rng, seg, w, h, min_len, max_len, used2)
            if seg is None:
                continue
        else:
            a = (seg["start_pos"]["x"], seg["start_pos"]["y"])
            b = (seg["end_pos"]["x"], seg["end_pos"]["y"])
            key = (a, b) if a <= b else (b, a)
            if key in used2:
                seg = _random_segment_within_range(rng, w, h, used2, min_len, max_len)
                if seg is None:
                    continue
        ok_walls.append(seg)
        if len(ok_walls) == count:
            break

    return ok_walls[:count]

# ---------- Map builder ----------
def _build_length_constrained_walls_with_connectivity(level: Dict[str, Any], rng: random.Random) -> List[Dict[str, Dict[str,int]]]:
    """Generate walls (respecting length bounds and count) but reject layouts that
       create sealed rooms. We accept only if the open area forms one component
       or the largest component is at least ~85% of all open cells."""
    w, h = level["width"], level["height"]
    min_len, max_len = _segment_len_bounds(w, h)

    def _norm_key(seg):
        a = (seg["start_pos"]["x"], seg["start_pos"]["y"])
        b = (seg["end_pos"]["x"], seg["end_pos"]["y"])
        return (a, b) if a <= b else (b, a)

    # Try multiple shuffles; gradually reduce the number of "kept" base segments if needed.
    for keep_target in [2, 1, 0, 2, 1, 0]:  # give extra chances
        for _ in range(40):
            base_walls = _deterministic_internal_walls(level)
            keep_n = min(keep_target, len(base_walls))
            keep = rng.sample(base_walls, keep_n) if base_walls else []
            used: Set[Tuple[Tuple[int,int], Tuple[int,int]]] = set(_norm_key(s) for s in keep)

            need = level["num_wall"] - keep_n
            gen_walls: List[Dict[str, Dict[str, int]]] = []
            for _i in range(max(0, need)):
                seg = _random_segment_within_range(rng, w, h, used, min_len, max_len)
                if seg:
                    gen_walls.append(seg)
            walls = keep + gen_walls

            # Final validation & dedupe with bounds
            validated = []
            used_final: Set[Tuple[Tuple[int,int], Tuple[int,int]]] = set()
            for seg in walls:
                L = _segment_cells_len(seg)
                if not (min_len <= L <= max_len):
                    seg = _refit_segment_to_range(rng, seg, w, h, min_len, max_len, used_final)
                    if seg is None:
                        continue
                else:
                    key = _norm_key(seg)
                    if key in used_final:
                        seg = _random_segment_within_range(rng, w, h, used_final, min_len, max_len)
                        if seg is None:
                            continue
                validated.append(seg)
            walls = validated[:level["num_wall"]]

            wall_cells = _collect_wall_cells(walls)
            total_open = w * h - len(wall_cells)
            comps = _all_free_components(w, h, wall_cells)
            if not comps:
                continue
            largest = max(len(c) for c in comps)
            # Accept if there is a single open component, or the largest one is big enough.
            if len(comps) == 1 or largest >= int(0.85 * total_open):
                return walls
    # Fallback to original pipeline if we couldn't satisfy the connectivity preference
    return _deterministic_internal_walls(level)

def build_map(level: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    width, height = level["width"], level["height"]

    # Build walls with the usual constraints, but prefer layouts without sealed rooms.
    walls = _build_length_constrained_walls_with_connectivity(level, rng)

    wall_cells = _collect_wall_cells(walls)

    # Player must not overlap walls and should start inside the largest open component.
    comps_largest = _largest_component(width, height, wall_cells)
    forbidden: Set[Tuple[int, int]] = set(wall_cells)

    if comps_largest:
        # Prefer interior cells of the largest component.
        interior = [(x, y) for (x, y) in comps_largest if 0 < x < width-1 and 0 < y < height-1 and (x, y) not in forbidden]
        if interior:
            px, py = rng.choice(interior)
        else:
            px, py = _random_free_cell(rng, width, height, forbidden)
    else:
        px, py = _random_free_cell(rng, width, height, forbidden)

    player_pos = {"x": px, "y": py}
    player_cell = (px, py)

    # Small obstacles: keep the existing behavior but require the player's reachable component
    # (with walls+obstacles) to remain reasonably large to avoid cramped starts.
    small_obstacles = _place_small_obstacles(rng, level, wall_cells, player_cell)

    # Enemies: avoid borders most of the time and spread them.
    enemies_pts = _place_enemies(rng, level, wall_cells, player_cell, small_obstacles)

    # Final safety checks (non-overlap)
    occupied = set(wall_cells) | {(player_pos["x"], player_pos["y"])}
    for o in small_obstacles:
        assert (o["x"], o["y"]) not in occupied
        occupied.add((o["x"], o["y"]))
    for e in enemies_pts:
        assert (e["x"], e["y"]) not in occupied
        occupied.add((e["x"], e["y"]))

    # Final reachability: every enemy reachable from player given walls+obstacles
    blocked = set(wall_cells) | {(o["x"], o["y"]) for o in small_obstacles}
    reach = _reachable_from(player_cell, width, height, blocked)
    assert all((e["x"], e["y"]) in reach for e in enemies_pts)

    return {
        "level": level,
        "walls": walls,
        "small_obstacles": small_obstacles,
        "enemies": enemies_pts,
        "player_pos": player_pos,
    }

# ---------- Parsing ----------
CL_RE = re.compile(r"current_level\s*=\s*(\d+)", re.IGNORECASE)
CD_RE = re.compile(r"current_difficulty\s*=\s*(\d+)", re.IGNORECASE)
LEVEL_OBJ_RE = re.compile(
    r"level\s*=\s*Level\s*\(\s*name\s*=\s*(?P<name>\d+)\s*,\s*difficulty\s*=\s*(?P<diff>\d+)\s*,\s*time\s*=\s*(?P<time>\d+)\s*,\s*width\s*=\s*(?P<width>\d+)\s*,\s*height\s*=\s*(?P<height>\d+)\s*,\s*num_wall\s*=\s*(?P<num_wall>\d+)\s*,\s*num_enemies\s*=\s*(?P<num_enemies>\d+)\s*\)",
    re.IGNORECASE | re.DOTALL,
)
DIFF_LINE_RE = re.compile(r"^\s*difficulty\s*=\s*(?P<d>\d+)\s*$", re.IGNORECASE | re.MULTILINE)
MODE_RE = re.compile(r"^\s*(create_next_level|create_next_map)\s*$", re.IGNORECASE | re.MULTILINE)
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

# ---------- Sanitizer ----------
def _sanitize_input_for_record(input_text: str) -> str:
    return re.sub(r"(?mi)^\s*seed\s*=\s*-?\d+\s*\n?", "", input_text).rstrip()

# ---------- Record builders ----------
def build_record(system_text: str, input_text: str, output_obj: Dict[str, Any]) -> Dict[str, Any]:
    output_json_str = json.dumps(output_obj, separators=(", ", ": "), sort_keys=False)
    return {"system": system_text, "input": input_text, "output": output_json_str}

def generate_next_level_record(input_text: str) -> Dict[str, Any]:
    current_level = parse_current_level(input_text)
    if current_level is None:
        raise ValueError("Could not parse current_level from input text.")
    # name N = current_level + 1, difficulty = current_level - 1
    difficulty = current_level - 1
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

    # Enforce dataset-specific small_obstacles counts:
    try:
        target = None
        if map_obj["level"]["name"] == 8:  target = 10
        elif map_obj["level"]["name"] == 9:  target = 11
        elif map_obj["level"]["name"] == 10: target = 12
        elif map_obj["level"]["name"] == 11: target = 13
        if target is not None:
            cur = map_obj.get("small_obstacles", [])
            if len(cur) > target:
                map_obj["small_obstacles"] = cur[:target]
            elif len(cur) < target:
                need = target - len(cur)
                walls_set = _collect_wall_cells(map_obj["walls"])
                forbidden = set(walls_set)
                forbidden.add((map_obj["player_pos"]["x"], map_obj["player_pos"]["y"]))
                for p in cur: forbidden.add((p["x"], p["y"]))
                for e in map_obj["enemies"]: forbidden.add((e["x"], e["y"]))
                fillers = _uniform_points(_rng_from_input(input_text), need, level_obj["width"], level_obj["height"], forbidden)
                map_obj["small_obstacles"].extend(fillers)
    except Exception:
        pass

    return build_record(SYSTEM_PROMPT, _sanitize_input_for_record(input_text), map_obj)

# ---------- Formatting helpers ----------
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

# ---------- Stage input builders (unchanged, through Stage 18) ----------
def build_stage3_input_from_map(next_difficulty: int,
                                prev_level: Dict[str, Any],
                                map_obj: Dict[str, Any]) -> str:
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
    lvl3_txt = _fmt_level_for_input(prev_level_for_map)
    lvl4_txt = _fmt_level_for_input(stage3_level_cfg)
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
    lvl3 = level3_map["level"]; lvl4 = level4_map["level"]
    lvl3_txt = _fmt_level_for_input(lvl3); lvl4_txt = _fmt_level_for_input(lvl4)
    map3_txt = _fmt_map_tiles(level3_map); map4_txt = _fmt_map_tiles(level4_map)
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
    lvl3 = level3_map["level"]; lvl4 = level4_map["level"]; lvl5 = stage5_level_cfg
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
    lvl3 = level3_map["level"]; lvl4 = level4_map["level"]; lvl5 = level5_map["level"]
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

def build_stage8_input_from_stage7(level3_map: Dict[str, Any],
                                   level4_map: Dict[str, Any],
                                   level5_map: Dict[str, Any],
                                   stage7_level_cfg: Dict[str, Any]) -> str:
    lvl3 = level3_map["level"]; lvl4 = level4_map["level"]; lvl5 = level5_map["level"]; lvl6 = stage7_level_cfg
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
        f"prev_levels=[{lvl3_txt}, {lvl4_txt}, {lvl5_txt}, {lvl6_txt}], "
        f"prev_level_maps=[{map3_txt}, {map4_txt}, {map5_txt}])\n"
        f"level = {lvl6_txt}"
    )
    return input_str

def build_stage9_input_from_stage8(level3_map: Dict[str, Any],
                                   level4_map: Dict[str, Any],
                                   level5_map: Dict[str, Any],
                                   level6_map: Dict[str, Any]) -> str:
    lvl4 = level4_map["level"]; lvl5 = level5_map["level"]; lvl6 = level6_map["level"]
    lvl4_txt = _fmt_level_for_input(lvl4)
    lvl5_txt = _fmt_level_for_input(lvl5)
    lvl6_txt = _fmt_level_for_input(lvl6)
    map4_txt = _fmt_map_tiles(level4_map)
    map5_txt = _fmt_map_tiles(level5_map)
    map6_txt = _fmt_map_tiles(level6_map)
    input_str = (
        "create_next_level\n\n"
        f"self = LevelManager(current_level={lvl6['name']}, current_difficulty={lvl5['difficulty']}, "
        f"prev_levels=[{lvl4_txt}, {lvl5_txt}, {lvl6_txt}], prev_level_maps=[{map4_txt}, {map5_txt}, {map6_txt}])\n"
        f"last_levels = [{lvl4_txt}, {lvl5_txt}, {lvl6_txt}]\n"
        f"difficulty = {lvl5['difficulty']}"
    )
    return input_str

def build_stage10_input_from_stage9(level3_map: Dict[str, Any],
                                    level4_map: Dict[str, Any],
                                    level5_map: Dict[str, Any],
                                    level6_map: Dict[str, Any],
                                    stage9_level_cfg: Dict[str, Any]) -> str:
    lvl4 = level4_map["level"]; lvl5 = level5_map["level"]; lvl6 = level6_map["level"]; lvl7 = stage9_level_cfg
    lvl4_txt = _fmt_level_for_input(lvl4)
    lvl5_txt = _fmt_level_for_input(lvl5)
    lvl6_txt = _fmt_level_for_input(lvl6)
    lvl7_txt = _fmt_level_for_input(lvl7)
    map4_txt = _fmt_map_tiles(level4_map)
    map5_txt = _fmt_map_tiles(level5_map)
    map6_txt = _fmt_map_tiles(level6_map)
    input_str = (
        "create_next_map\n\n"
        f"self = LevelManager(current_level={lvl6['name']}, current_difficulty={lvl5['difficulty']}, "
        f"prev_levels=[{lvl4_txt}, {lvl5_txt}, {lvl6_txt}, {lvl7_txt}], "
        f"prev_level_maps=[{map4_txt}, {map5_txt}, {map6_txt}])\n"
        f"level = {lvl7_txt}"
    )
    return input_str

def build_stage11_input_from_stage10(level5_map: Dict[str, Any],
                                     level6_map: Dict[str, Any],
                                     level7_map: Dict[str, Any]) -> str:
    lvl5 = level5_map["level"]; lvl6 = level6_map["level"]; lvl7 = level7_map["level"]
    lvl5_txt = _fmt_level_for_input(lvl5)
    lvl6_txt = _fmt_level_for_input(lvl6)
    lvl7_txt = _fmt_level_for_input(lvl7)
    map5_txt = _fmt_map_tiles(level5_map)
    map6_txt = _fmt_map_tiles(level6_map)
    map7_txt = _fmt_map_tiles(level7_map)
    input_str = (
        "create_next_level\n\n"
        f"self = LevelManager(current_level={lvl7['name']}, current_difficulty={lvl6['difficulty']}, "
        f"prev_levels=[{lvl5_txt}, {lvl6_txt}, {lvl7_txt}], prev_level_maps=[{map5_txt}, {map6_txt}, {map7_txt}])\n"
        f"last_levels = [{lvl5_txt}, {lvl6_txt}, {lvl7_txt}]\n"
        f"difficulty = {lvl6['difficulty']}"
    )
    return input_str

def build_stage12_input_from_stage11(level5_map: Dict[str, Any],
                                     level6_map: Dict[str, Any],
                                     level7_map: Dict[str, Any],
                                     stage11_level_cfg: Dict[str, Any]) -> str:
    lvl5 = level5_map["level"]; lvl6 = level6_map["level"]; lvl7 = level7_map["level"]; lvl8 = stage11_level_cfg
    lvl5_txt = _fmt_level_for_input(lvl5)
    lvl6_txt = _fmt_level_for_input(lvl6)
    lvl7_txt = _fmt_level_for_input(lvl7)
    lvl8_txt = _fmt_level_for_input(lvl8)
    map5_txt = _fmt_map_tiles(level5_map)
    map6_txt = _fmt_map_tiles(level6_map)
    map7_txt = _fmt_map_tiles(level7_map)
    input_str = (
        "create_next_map\n\n"
        f"self = LevelManager(current_level={lvl7['name']}, current_difficulty={lvl6['difficulty']}, "
        f"prev_levels=[{lvl5_txt}, {lvl6_txt}, {lvl7_txt}, {lvl8_txt}], "
        f"prev_level_maps=[{map5_txt}, {map6_txt}, {map7_txt}])\n"
        f"level = {lvl8_txt}"
    )
    return input_str

def build_stage13_input_from_stage12(level6_map: Dict[str, Any],
                                     level7_map: Dict[str, Any],
                                     level8_map: Dict[str, Any]) -> str:
    lvl6 = level6_map["level"]; lvl7 = level7_map["level"]; lvl8 = level8_map["level"]
    lvl6_txt = _fmt_level_for_input(lvl6)
    lvl7_txt = _fmt_level_for_input(lvl7)
    lvl8_txt = _fmt_level_for_input(lvl8)
    map6_txt = _fmt_map_tiles(level6_map)
    map7_txt = _fmt_map_tiles(level7_map)
    map8_txt = _fmt_map_tiles(level8_map)
    input_str = (
        "create_next_level\n\n"
        f"self = LevelManager(current_level={lvl8['name']}, current_difficulty={lvl7['difficulty']}, "
        f"prev_levels=[{lvl6_txt}, {lvl7_txt}, {lvl8_txt}], "
        f"prev_level_maps=[{map6_txt}, {map7_txt}, {map8_txt}])\n"
        f"last_levels = [{lvl6_txt}, {lvl7_txt}, {lvl8_txt}]\n"
        f"difficulty = {lvl7['difficulty']}"
    )
    return input_str

def build_stage14_input_from_stage13(level6_map: Dict[str, Any],
                                     level7_map: Dict[str, Any],
                                     level8_map: Dict[str, Any],
                                     stage13_level_cfg: Dict[str, Any]) -> str:
    lvl6 = level6_map["level"]; lvl7 = level7_map["level"]; lvl8 = level8_map["level"]; lvl9 = stage13_level_cfg
    lvl6_txt = _fmt_level_for_input(lvl6)
    lvl7_txt = _fmt_level_for_input(lvl7)
    lvl8_txt = _fmt_level_for_input(lvl8)
    lvl9_txt = _fmt_level_for_input(lvl9)
    map6_txt = _fmt_map_tiles(level6_map)
    map7_txt = _fmt_map_tiles(level7_map)
    map8_txt = _fmt_map_tiles(level8_map)
    input_str = (
        "create_next_map\n\n"
        f"self = LevelManager(current_level={lvl8['name']}, current_difficulty={lvl7['difficulty']}, "
        f"prev_levels=[{lvl6_txt}, {lvl7_txt}, {lvl8_txt}, {lvl9_txt}], "
        f"prev_level_maps=[{map6_txt}, {map7_txt}, {map8_txt}])\n"
        f"level = {lvl9_txt}"
    )
    return input_str

def build_stage15_input_from_stage14(level7_map: Dict[str, Any],
                                     level8_map: Dict[str, Any],
                                     level9_map: Dict[str, Any]) -> str:
    lvl7 = level7_map["level"]; lvl8 = level8_map["level"]; lvl9 = level9_map["level"]
    lvl7_txt = _fmt_level_for_input(lvl7)
    lvl8_txt = _fmt_level_for_input(lvl8)
    lvl9_txt = _fmt_level_for_input(lvl9)
    map7_txt = _fmt_map_tiles(level7_map)
    map8_txt = _fmt_map_tiles(level8_map)
    map9_txt = _fmt_map_tiles(level9_map)
    input_str = (
        "create_next_level\n\n"
        f"self = LevelManager(current_level={lvl9['name']}, current_difficulty={lvl8['difficulty']}, "
        f"prev_levels=[{lvl7_txt}, {lvl8_txt}, {lvl9_txt}], prev_level_maps=[{map7_txt}, {map8_txt}, {map9_txt}])\n"
        f"last_levels = [{lvl7_txt}, {lvl8_txt}, {lvl9_txt}]\n"
        f"difficulty = {lvl8['difficulty']}"
    )
    return input_str

def build_stage16_input_from_stage15(level7_map: Dict[str, Any],
                                     level8_map: Dict[str, Any],
                                     level9_map: Dict[str, Any],
                                     stage15_level_cfg: Dict[str, Any]) -> str:
    lvl7 = level7_map["level"]; lvl8 = level8_map["level"]; lvl9 = level9_map["level"]; lvl10 = stage15_level_cfg
    lvl7_txt = _fmt_level_for_input(lvl7)
    lvl8_txt = _fmt_level_for_input(lvl8)
    lvl9_txt = _fmt_level_for_input(lvl9)
    lvl10_txt = _fmt_level_for_input(lvl10)
    map7_txt = _fmt_map_tiles(level7_map)
    map8_txt = _fmt_map_tiles(level8_map)
    map9_txt = _fmt_map_tiles(level9_map)
    input_str = (
        "create_next_map\n\n"
        f"self = LevelManager(current_level={lvl9['name']}, current_difficulty={lvl8['difficulty']}, "
        f"prev_levels=[{lvl7_txt}, {lvl8_txt}, {lvl9_txt}, {lvl10_txt}], "
        f"prev_level_maps=[{map7_txt}, {map8_txt}, {map9_txt}])\n"
        f"level = {lvl10_txt}"
    )
    return input_str

def build_stage17_input_from_stage16(level8_map: Dict[str, Any],
                                     level9_map: Dict[str, Any],
                                     level10_map: Dict[str, Any]) -> str:
    lvl8 = level8_map["level"]; lvl9 = level9_map["level"]; lvl10 = level10_map["level"]
    lvl8_txt = _fmt_level_for_input(lvl8)
    lvl9_txt = _fmt_level_for_input(lvl9)
    lvl10_txt = _fmt_level_for_input(lvl10)
    map8_txt = _fmt_map_tiles(level8_map)
    map9_txt = _fmt_map_tiles(level9_map)
    map10_txt = _fmt_map_tiles(level10_map)
    input_str = (
        "create_next_level\n\n"
        f"self = LevelManager(current_level={lvl10['name']}, current_difficulty={lvl9['difficulty']}, "
        f"prev_levels=[{lvl8_txt}, {lvl9_txt}, {lvl10_txt}], "
        f"prev_level_maps=[{map8_txt}, {map9_txt}, {map10_txt}])\n"
        f"last_levels = [{lvl8_txt}, {lvl9_txt}, {lvl10_txt}]\n"
        f"difficulty = {lvl9['difficulty']}"
    )
    return input_str

def build_stage18_input_from_stage17(level8_map: Dict[str, Any],
                                     level9_map: Dict[str, Any],
                                     level10_map: Dict[str, Any],
                                     stage17_level_cfg: Dict[str, Any]) -> str:
    lvl8 = level8_map["level"]; lvl9 = level9_map["level"]; lvl10 = level10_map["level"]; lvl11 = stage17_level_cfg
    lvl8_txt = _fmt_level_for_input(lvl8)
    lvl9_txt = _fmt_level_for_input(lvl9)
    lvl10_txt = _fmt_level_for_input(lvl10)
    lvl11_txt = _fmt_level_for_input(lvl11)
    map8_txt = _fmt_map_tiles(level8_map)
    map9_txt = _fmt_map_tiles(level9_map)
    map10_txt = _fmt_map_tiles(level10_map)
    input_str = (
        "create_next_map\n\n"
        f"self = LevelManager(current_level={lvl10['name']}, current_difficulty={lvl9['difficulty']}, "
        f"prev_levels=[{lvl8_txt}, {lvl9_txt}, {lvl10_txt}, {lvl11_txt}], "
        f"prev_level_maps=[{map8_txt}, {map9_txt}, {map10_txt}])\n"
        f"level = {lvl11_txt}"
    )
    return input_str

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

# ---------- Demo: Stage 1 → 18 (up to Level 11) ----------
if __name__ == "__main__":
    # Stage 1 (Level 3 config)
    stage1_input = (
        "create_next_level\n\n"
        "self = LevelManager(current_level=2, current_difficulty=1, prev_levels=[], prev_level_maps=[])\n"
        "last_levels = []\n"
        "difficulty = 1"
    )
    rec1 = generate_record(stage1_input)
    print(json.dumps(rec1, ensure_ascii=False))

    # Stage 2 (Level 3 map)
    stage2_input = (
        "create_next_map\n\n"
        "self = LevelManager(current_level=2, current_difficulty=1, prev_levels=[Level(name=3, difficulty=1, time=300, width=10, height=10, num_wall=5, num_enemies=2)], prev_level_maps=[])\n"
        "level = Level(name=3, difficulty=1, time=300, width=10, height=10, num_wall=5, num_enemies=2)"
    )
    rec2 = generate_record(stage2_input)
    print(json.dumps(rec2, ensure_ascii=False))

    # Stage 3 (Level 4 config)
    stage2_out = json.loads(rec2["output"])
    prev_level = stage2_out["level"]
    next_difficulty = 2
    stage3_input = build_stage3_input_from_map(next_difficulty, prev_level, stage2_out)
    rec3 = generate_record(stage3_input)
    print(json.dumps(rec3, ensure_ascii=False))

    # Stage 4 (Level 4 map)
    stage3_level_cfg = json.loads(rec3["output"])
    rec4 = generate_record(build_stage4_input_from_stage3(prev_level_for_map=prev_level, stage3_level_cfg=stage3_level_cfg, stage2_map_obj=stage2_out))
    print(json.dumps(rec4, ensure_ascii=False))

    # Stage 5 (Level 5 config)
    stage4_out = json.loads(rec4["output"])
    rec5 = generate_record(build_stage5_input_from_stage4(stage2_out, stage4_out))
    print(json.dumps(rec5, ensure_ascii=False))

    # Stage 6 (Level 5 map)
    stage5_level_cfg = json.loads(rec5["output"])
    rec6 = generate_record(build_stage6_input_from_stage5(stage2_out, stage4_out, stage5_level_cfg))
    print(json.dumps(rec6, ensure_ascii=False))

    # Stage 7 (Level 6 config)
    stage6_out = json.loads(rec6["output"])
    rec7 = generate_record(build_stage7_input_from_stage6(stage2_out, stage4_out, stage6_out))
    print(json.dumps(rec7, ensure_ascii=False))

    # Stage 8 (Level 6 map)
    stage7_level_cfg = json.loads(rec7["output"])
    rec8 = generate_record(build_stage8_input_from_stage7(stage2_out, stage4_out, stage6_out, stage7_level_cfg))
    print(json.dumps(rec8, ensure_ascii=False))

    # Stage 9 (Level 7 config)
    stage8_out = json.loads(rec8["output"])
    rec9 = generate_record(build_stage9_input_from_stage8(stage2_out, stage4_out, stage6_out, stage8_out))
    print(json.dumps(rec9, ensure_ascii=False))

    # Stage 10 (Level 7 map)
    stage9_level_cfg = json.loads(rec9["output"])
    rec10 = generate_record(build_stage10_input_from_stage9(stage2_out, stage4_out, stage6_out, stage8_out, stage9_level_cfg))
    print(json.dumps(rec10, ensure_ascii=False))

    # Stage 11 (Level 8 config)
    stage10_out = json.loads(rec10["output"])  # Level 7 map
    rec11 = generate_record(build_stage11_input_from_stage10(level5_map=stage6_out, level6_map=stage8_out, level7_map=stage10_out))
    print(json.dumps(rec11, ensure_ascii=False))

    # Stage 12 (Level 8 map; small_obstacles must be 10)
    stage11_level_cfg = json.loads(rec11["output"])
    rec12 = generate_record(build_stage12_input_from_stage11(level5_map=stage6_out, level6_map=stage8_out, level7_map=stage10_out, stage11_level_cfg=stage11_level_cfg))
    print(json.dumps(rec12, ensure_ascii=False))

    # Stage 13 (Level 9 config)
    stage12_out = json.loads(rec12["output"])  # Level 8 map
    rec13 = generate_record(build_stage13_input_from_stage12(level6_map=stage8_out, level7_map=stage10_out, level8_map=stage12_out))
    print(json.dumps(rec13, ensure_ascii=False))

    # Stage 14 (Level 9 map; small_obstacles must be 11)
    stage13_level_cfg = json.loads(rec13["output"])
    rec14 = generate_record(build_stage14_input_from_stage13(level6_map=stage8_out, level7_map=stage10_out, level8_map=stage12_out, stage13_level_cfg=stage13_level_cfg))
    print(json.dumps(rec14, ensure_ascii=False))

    # Stage 15 (Level 10 config)
    stage14_out = json.loads(rec14["output"])  # Level 9 map
    rec15 = generate_record(build_stage15_input_from_stage14(level7_map=stage10_out, level8_map=stage12_out, level9_map=stage14_out))
    print(json.dumps(rec15, ensure_ascii=False))

    # Stage 16 (Level 10 map; small_obstacles must be 12)
    stage15_level_cfg = json.loads(rec15["output"])
    rec16 = generate_record(build_stage16_input_from_stage15(level7_map=stage10_out, level8_map=stage12_out, level9_map=stage14_out, stage15_level_cfg=stage15_level_cfg))
    print(json.dumps(rec16, ensure_ascii=False))

    # Stage 17 (Level 11 config)
    stage16_out = json.loads(rec16["output"])  # Level 10 map
    rec17 = generate_record(build_stage17_input_from_stage16(level8_map=stage12_out, level9_map=stage14_out, level10_map=stage16_out))
    print(json.dumps(rec17, ensure_ascii=False))

    # Stage 18 (Level 11 map; small_obstacles must be 13)
    stage17_level_cfg = json.loads(rec17["output"])
    rec18 = generate_record(build_stage18_input_from_stage17(level8_map=stage12_out, level9_map=stage14_out, level10_map=stage16_out, stage17_level_cfg=stage17_level_cfg))
    print(json.dumps(rec18, ensure_ascii=False))
