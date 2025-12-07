import sys, os, re, json, argparse
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image, ImageDraw
import pygame

# ---------- Appearance / Defaults ----------
TILE = 32
BG_COLOR = (235, 235, 235)
GRID_COLOR = (200, 200, 200)
BLOCK_COLOR = (80, 80, 80)
ENEMY_COLOR = (200, 50, 50)
PLAYER_COLOR = (50, 100, 220)
DRAW_GRID = True

# ---------- IO Helpers ----------
def read_text_guessing_encoding(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="ignore")

def iter_jsonl(path: str):
    """
    Yields JSON objects from a JSONL file. Lines that fail JSON parsing are skipped.
    """
    text = read_text_guessing_encoding(path)
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            print(f"[WARN] Skipping line {i}: not valid JSON")
            continue

# ---------- Map Extraction ----------
def looks_like_map_obj(obj: Dict[str, Any]) -> bool:
    needed = ("level", "walls", "small_obstacles", "enemies", "player_pos")
    return isinstance(obj, dict) and all(k in obj for k in needed)

def _extract_map_from_any(blob: Any) -> Optional[Dict[str, Any]]:
    """
    Robustly find a map dict in:
      - a dict with map keys
      - a dict that has 'output' which is a dict or a JSON string
      - a JSON string that decodes into either of the above
    """
    if isinstance(blob, dict):
        if looks_like_map_obj(blob):
            return blob
        if "output" in blob:
            out = blob["output"]
            # if output is a JSON string, decode it
            if isinstance(out, str):
                try:
                    out = json.loads(out)
                except Exception:
                    return None
            return _extract_map_from_any(out)
        return None

    if isinstance(blob, str):
        try:
            parsed = json.loads(blob)
        except Exception:
            return None
        return _extract_map_from_any(parsed)

    return None

def extract_maps_from_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Reads the JSONL and returns a list of map dicts.
    Works with your format where `output` is a JSON-encoded string.
    Ignores 'create_next_level' lines automatically.
    """
    maps: List[Dict[str, Any]] = []
    for rec in iter_jsonl(path):
        m = _extract_map_from_any(rec)
        if m is not None:
            maps.append(m)
    return maps

# ---------- Map Rendering (Pillow) ----------
def draw_map_image(rows: List[str]) -> Image.Image:
    """
    Render an ASCII list of rows to a crisp PNG (no blur).
    """
    h, w = len(rows), len(rows[0])
    img = Image.new("RGB", (w * TILE, h * TILE), BG_COLOR)
    d = ImageDraw.Draw(img)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            x0, y0 = x * TILE, y * TILE
            # floor
            d.rectangle([x0, y0, x0 + TILE - 1, y0 + TILE - 1], fill=BG_COLOR)
            # content
            if ch in ("B", "#"):
                d.rectangle([x0, y0, x0 + TILE - 1, y0 + TILE - 1], fill=BLOCK_COLOR)
            elif ch == "E":
                r = TILE // 2 - 2
                d.ellipse([x0 + TILE // 2 - r, y0 + TILE // 2 - r,
                           x0 + TILE // 2 + r, y0 + TILE // 2 + r], fill=ENEMY_COLOR)
            elif ch == "P":
                r = TILE // 2 - 2
                d.ellipse([x0 + TILE // 2 - r, y0 + TILE // 2 - r,
                           x0 + TILE // 2 + r, y0 + TILE // 2 + r], fill=PLAYER_COLOR)
            # grid
            if DRAW_GRID:
                d.rectangle([x0, y0, x0 + TILE - 1, y0 + TILE - 1], outline=GRID_COLOR, width=1)
    return img

def map_to_ascii(m: Dict[str, Any], add_border: bool = True) -> List[str]:
    """
    Build ASCII tile rows from a map dict.
      '.' floor, '#' walls, 'B' small_obstacles, 'E' enemies, 'P' player.
    Optionally adds a 1-tile 'B' border to match earlier previews.
    """
    lvl = m["level"]
    w = int(lvl["width"])
    h = int(lvl["height"])

    tiles = [["." for _ in range(w)] for _ in range(h)]

    # walls (assume H/V)
    for wl in m["walls"]:
        sx, sy = int(wl["start_pos"]["x"]), int(wl["start_pos"]["y"])
        ex, ey = int(wl["end_pos"]["x"]), int(wl["end_pos"]["y"])
        if sx == ex:
            y1, y2 = sorted((sy, ey))
            for y in range(y1, y2 + 1):
                if 0 <= y < h and 0 <= sx < w:
                    tiles[y][sx] = "#"
        elif sy == ey:
            x1, x2 = sorted((sx, ex))
            for x in range(x1, x2 + 1):
                if 0 <= sy < h and 0 <= x < w:
                    tiles[sy][x] = "#"

    # obstacles
    for p in m.get("small_obstacles", []):
        x, y = int(p["x"]), int(p["y"])
        if 0 <= y < h and 0 <= x < w:
            tiles[y][x] = "B"

    # enemies
    for e in m.get("enemies", []):
        x, y = int(e["x"]), int(e["y"])
        if 0 <= y < h and 0 <= x < w:
            tiles[y][x] = "E"

    # player
    px, py = int(m["player_pos"]["x"]), int(m["player_pos"]["y"])
    if 0 <= py < h and 0 <= px < w:
        tiles[py][px] = "P"

    rows = ["".join(row) for row in tiles]

    if not add_border:
        return rows

    # add 1-tile 'B' border
    top = "B" * (w + 2)
    bordered = [top]
    for r in rows:
        bordered.append("B" + r + "B")
    bordered.append(top)
    return bordered

# ---------- Export ----------
def sanitize_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem)

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def export_maps_to_png(maps: List[Dict[str, Any]], in_path: str, out_dir: str, limit: Optional[int] = None) -> List[str]:
    """
    Converts each map to ASCII and saves a PNG image. Returns list of file paths.
    """
    ensure_outdir(out_dir)
    stem = sanitize_stem(in_path)
    saved = []
    n = len(maps) if limit is None else min(limit, len(maps))
    for i in range(n):
        m = maps[i]
        rows = map_to_ascii(m, add_border=True)
        img = draw_map_image(rows)
        fname = f"{stem}_map_{i+1:04d}.png"
        fpath = os.path.join(out_dir, fname)
        img.save(fpath, "PNG")
        saved.append(fpath)
    return saved

# ---------- Viewer (pygame) ----------
def view_images(paths: List[str]):
    """
    Simple viewer: load PNGs with pygame and flip through using left/right arrows.
    ESC or window close to exit.
    """
    if not paths:
        print("[INFO] No images to view.")
        return
    pygame.init()
    idx = 0
    surf_cache: Dict[int, pygame.Surface] = {}

    def load_surface(i: int) -> pygame.Surface:
        if i in surf_cache:
            return surf_cache[i]
        img = pygame.image.load(paths[i])
        surf_cache[i] = img
        return img

    # start with first image size
    first = load_surface(0)
    screen = pygame.display.set_mode(first.get_size())
    pygame.display.set_caption(f"Map Viewer (1/{len(paths)}) - {os.path.basename(paths[0])}")

    clock = pygame.time.Clock()
    running = True

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key in (pygame.K_RIGHT, pygame.K_d):
                    idx = (idx + 1) % len(paths)
                    img = load_surface(idx)
                    pygame.display.set_mode(img.get_size())
                    pygame.display.set_caption(f"Map Viewer ({idx+1}/{len(paths)}) - {os.path.basename(paths[idx])}")
                elif e.key in (pygame.K_LEFT, pygame.K_a):
                    idx = (idx - 1) % len(paths)
                    img = load_surface(idx)
                    pygame.display.set_mode(img.get_size())
                    pygame.display.set_caption(f"Map Viewer ({idx+1}/{len(paths)}) - {os.path.basename(paths[idx])}")

        screen = pygame.display.get_surface()
        screen.blit(load_surface(idx), (0, 0))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Extract, export, and optionally view maps from a JSONL LLM log.")
    ap.add_argument("jsonl", help="Path to JSONL file (e.g., llm_io_log.jsonl)")
    ap.add_argument("--outdir", default="exported_maps", help="Output folder for PNGs")
    ap.add_argument("--limit", type=int, default=None, help="Max number of maps to process")
    ap.add_argument("--view", action="store_true", help="Open a viewer to flip through exported maps")
    return ap.parse_args()

def main():
    args = parse_args()
    maps = extract_maps_from_jsonl(args.jsonl)
    if not maps:
        print("[INFO] No map outputs found in JSONL.")
        sys.exit(0)

    if args.limit is not None:
        print(f"[INFO] Found {len(maps)} maps; exporting first {args.limit}.")
    else:
        print(f"[INFO] Found {len(maps)} maps; exporting all.")

    saved = export_maps_to_png(maps, args.jsonl, args.outdir, limit=args.limit)
    print(f"[OK] Saved {len(saved)} PNG(s) to {args.outdir}")

    if args.view:
        view_images(saved)

if __name__ == "__main__":
    main()
