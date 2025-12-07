# Pattern_Validator.py

import sys
import json
import argparse
import re
from typing import Any, Dict, List, Tuple, Optional


LEVEL_STRICT_RE = re.compile(
    r'^\{"name":\d+,"difficulty":\d+,"time":\d+,"width":\d+,"height":\d+,"num_wall":\d+,"num_enemies":\d+\}$'
)

POS_STRICT = r'\{"x":\d+,"y":\d+\}'
WALL_STRICT = r'\{"start_pos":' + POS_STRICT + r',"end_pos":' + POS_STRICT + r'\}'
ARR_OF_POS_STRICT = r'\[' + r'(?:' + POS_STRICT + r'(?:,' + POS_STRICT + r')*)?' + r'\]'
ARR_OF_WALLS_STRICT = r'\[' + r'(?:' + WALL_STRICT + r'(?:,' + WALL_STRICT + r')*)?' + r'\]'

LEVEL_INLINE_STRICT = (
    r'\{"name":\d+,"difficulty":\d+,"time":\d+,"width":\d+,"height":\d+,"num_wall":\d+,"num_enemies":\d+\}'
)
MAP_STRICT_RE = re.compile(
    r'^\{"level":' + LEVEL_INLINE_STRICT +
    r',"walls":' + ARR_OF_WALLS_STRICT +
    r',"small_obstacles":' + ARR_OF_POS_STRICT +
    r',"enemies":' + ARR_OF_POS_STRICT +
    r',"player_pos":' + POS_STRICT +
    r'\}$'
)


# JSON structure checks 

def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)

def _is_pos(d: Any) -> bool:
    return isinstance(d, dict) and set(d.keys()) == {"x", "y"} and _is_int(d.get("x")) and _is_int(d.get("y"))

def _is_level(d: Any) -> Tuple[bool, Optional[str]]:
    if not isinstance(d, dict):
        return False, "Level is not a JSON object"
    required = ["name","difficulty","time","width","height","num_wall","num_enemies"]
    if set(d.keys()) != set(required):
        return False, f"Level keys mismatch; got {list(d.keys())}"
    for k in required:
        if not _is_int(d[k]):
            return False, f"Level '{k}' must be an integer"
    return True, None

def _is_wall(d: Any) -> Tuple[bool, Optional[str]]:
    if not isinstance(d, dict) or set(d.keys()) != {"start_pos", "end_pos"}:
        return False, "Wall must have exactly keys: start_pos, end_pos"
    if not _is_pos(d["start_pos"]):
        return False, "Wall.start_pos must be {\"x\":int,\"y\":int}"
    if not _is_pos(d["end_pos"]):
        return False, "Wall.end_pos must be {\"x\":int,\"y\":int}"
    return True, None

def _is_map(d: Any) -> Tuple[bool, Optional[str]]:
    if not isinstance(d, dict):
        return False, "Map is not a JSON object"
    needed = ["level", "walls", "small_obstacles", "enemies", "player_pos"]
    if set(d.keys()) != set(needed):
        return False, f"Map keys mismatch; got {list(d.keys())}"

    ok, err = _is_level(d.get("level"))
    if not ok:
        return False, f"level invalid: {err}"

    walls = d.get("walls")
    if not isinstance(walls, list):
        return False, "walls must be an array"
    for i, w in enumerate(walls):
        ok, err = _is_wall(w)
        if not ok:
            return False, f"walls[{i}] invalid: {err}"

    small = d.get("small_obstacles")
    if not isinstance(small, list):
        return False, "small_obstacles must be an array"
    for i, p in enumerate(small):
        if not _is_pos(p):
            return False, f"small_obstacles[{i}] invalid: must be {{\"x\":int,\"y\":int}}"

    enemies = d.get("enemies")
    if not isinstance(enemies, list):
        return False, "enemies must be an array"
    for i, p in enumerate(enemies):
        if not _is_pos(p):
            return False, f"enemies[{i}] invalid: must be {{\"x\":int,\"y\":int}}"

    if not _is_pos(d.get("player_pos")):
        return False, "player_pos must be {\"x\":int,\"y\":int}"

    return True, None


# Helpers to unwrap nested/escaped JSON

def _unwrap_stringified_json(text: str) -> Tuple[Optional[Any], Optional[str]]:
    """
    Try to parse 'text' as JSON. If the result is a string that itself looks like JSON,
    try to parse that too (up to two times). Returns (value, error_message_or_None).
    """
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"Not valid JSON: {e}"

    depth = 0
    while isinstance(value, str) and depth < 2:
        s = value.strip()
        if s.startswith("{") or s.startswith("["):
            try:
                value = json.loads(s)
            except json.JSONDecodeError as e:
                return None, f"Inner string not valid JSON: {e}"
            depth += 1
        else:
            break
    return value, None


# Public functions used by the CLI

def _minified(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

def is_level_strict_minified(obj: Dict[str, Any]) -> bool:
    return LEVEL_STRICT_RE.match(_minified(obj)) is not None

def is_map_strict_minified(obj: Dict[str, Any]) -> bool:
    return MAP_STRICT_RE.match(_minified(obj)) is not None

def validate_text(raw_text: str) -> Dict[str, Any]:
    """
    Accept a raw line of text. It may be:
      - a JSON object,
      - or a JSON string that contains an object.
    We unwrap if needed, then check whether it matches the Level or Map schema.
    """
    obj, err = _unwrap_stringified_json(raw_text)
    if err:
        return {"kind": "INVALID", "strict": False, "reason": err}

    if not isinstance(obj, dict):
        return {"kind": "INVALID", "strict": False, "reason": f"Top-level value is {type(obj).__name__}, expected object"}

    ok, why = _is_level(obj)
    if ok:
        strict = is_level_strict_minified(obj)
        out = {"kind": "LEVEL", "strict": strict}
        if not strict:
            out["note"] = "Structurally valid but not minified/ordered exactly."
            out["minified"] = _minified(obj)
        return out

    ok, why = _is_map(obj)
    if ok:
        strict = is_map_strict_minified(obj)
        out = {"kind": "MAP", "strict": strict}
        if not strict:
            out["note"] = "Structurally valid but not minified/ordered exactly."
            out["minified"] = _minified(obj)
        return out

    return {"kind": "INVALID", "strict": False, "reason": f"Object is neither LEVEL nor MAP: {why}"}


# Input handling

def load_records_from_text(text: str) -> List[str]:
    """
    Supports three input styles:
      1) JSONL: one JSON value per line (each line can also be a stringified JSON).
      2) A single JSON value.
      3) A JSON array of values.
    Returns a list of raw strings to pass to validate_text.
    """
    s = text.strip()
    if not s:
        return []

    if "\n" in s:
        return [ln.strip() for ln in s.splitlines() if ln.strip()]

    return [s]


# Command line entry point

def main():
    parser = argparse.ArgumentParser(
        description="Check if input lines are valid Level/Map JSON (handles stringified JSON too)."
    )
    parser.add_argument("--file", "-f", dest="file", default=None,
                        help="Path to a JSON/JSONL file. If missing, read from STDIN.")
    args = parser.parse_args()

    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as fh:
                text = fh.read()
        except FileNotFoundError:
            print(f"error: file not found: {args.file}")
            sys.exit(2)
    else:
        if sys.stdin.isatty():
            print('No --file given and no data piped on STDIN.')
            print('Example: python pattern_validator.py --file "C:\\Users\\123\\l6vl.jsonl"')
            sys.exit(1)
        text = sys.stdin.read()

    records = load_records_from_text(text)
    if not records:
        print("No input records found.")
        sys.exit(1)

    for idx, rec in enumerate(records, 1):
        result = validate_text(rec)
        kind = result["kind"]
        strict = result.get("strict", False)
        print(f"[{idx}] kind={kind} strict={strict}")
        if kind == "INVALID" and "reason" in result:
            print("reason:", result["reason"])
        if "note" in result:
            print("note:", result["note"])
        if "minified" in result:
            print("minified:")
            print(result["minified"])

if __name__ == "__main__":
    main()
