# new_validate.py
import sys
import json
import argparse
from collections import deque
from typing import List, Tuple


def validate_output_json(output_value):
    """
    Validate a single map output.

    - Treat wall segments as blocked cells (mark every cell on the segment).
    - Add small_obstacles to blocked.
    - Enforce enemy-count sanity:
        * NO enemies -> invalid
        * Fewer enemies than level.num_enemies -> invalid
      (We allow more than level.num_enemies, since the user asked to catch "no" or "less".)
    - BFS 4-directionally from the player's position across non-blocked cells.
    - Check if all enemies are in the visited set.
    """
    # Accept either a JSON string or a parsed dict
    data = json.loads(output_value) if isinstance(output_value, str) else output_value

    width = data["level"]["width"]
    height = data["level"]["height"]
    expected_enemies = data["level"].get("num_enemies", 0)

    enemies = data.get("enemies", [])
    actual_enemies = len(enemies)

    # Enemy-count checks
    enemy_count_ok = True
    enemy_count_error = None
    if actual_enemies == 0:
        enemy_count_ok = False
        enemy_count_error = f"no_enemies expected>= {expected_enemies}"
    elif actual_enemies < expected_enemies:
        enemy_count_ok = False
        enemy_count_error = f"too_few_enemies expected>= {expected_enemies} got={actual_enemies}"

    # Build blocked set (walls + small_obstacles)
    blocked = set()
    for wall in data.get("walls", []):
        x1, y1 = wall["start_pos"]["x"], wall["start_pos"]["y"]
        x2, y2 = wall["end_pos"]["x"], wall["end_pos"]["y"]

        if x1 == x2:  # vertical wall
            for y in range(min(y1, y2), max(y1, y2) + 1):
                blocked.add((x1, y))
        elif y1 == y2:  # horizontal wall
            for x in range(min(x1, x2), max(x1, x2) + 1):
                blocked.add((x, y1))
        else:
            # Ignore non-axis-aligned walls
            pass

    for obs in data.get("small_obstacles", []):
        blocked.add((obs["x"], obs["y"]))

    # BFS
    player = data["player_pos"]
    queue = deque()
    visited = set()
    start = (player["x"], player["y"])
    queue.append(start)
    visited.add(start)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in blocked and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))

    # Check if all enemies are reachable (only meaningful if there are enemies)
    all_reachable = True
    unreachable_enemies = []
    for enemy in enemies:
        pos = (enemy["x"], enemy["y"])
        if pos not in visited:
            all_reachable = False
            unreachable_enemies.append(pos)

    return {
        "level": data["level"]["name"],
        "all_reachable": all_reachable,
        "reachable_count": actual_enemies - len(unreachable_enemies),
        "total_enemies": actual_enemies,
        "unreachable_enemies": unreachable_enemies,
        "enemy_count_ok": enemy_count_ok,
        "enemy_count_error": enemy_count_error,
        "expected_enemies": expected_enemies,
        "actual_enemies": actual_enemies,
    }


def read_all_lines(path: str) -> List[Tuple[int, str]]:
    """Read file into list of (1-based line_number, raw_line_without_newline)."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            out.append((idx, line.rstrip("\n")))
    return out


def process_batches(lines: List[Tuple[int, str]], out_path: str, batch_size: int = 18) -> None:
    """
    Iterate through the JSONL in fixed-size batches (default 18 lines per batch).

    Keep (write) a batch to out_path if and only if:
    - Every line in the batch that is a "create_next_map" record validates YES.
    - Any parse/validation failure counts as NO for that map line, dropping the whole batch.

    Non-map lines in a kept batch are written unchanged (so 18 lines are preserved as-is).
    """
    wrote_any = False
    with open(out_path, "w", encoding="utf-8") as out_f:
        total_batches = (len(lines) + batch_size - 1) // batch_size
        for b in range(total_batches):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, len(lines))
            batch = lines[start_idx:end_idx]

            batch_ok = True
            batch_first_line = batch[0][0]
            batch_last_line = batch[-1][0]

            for (global_ln, raw) in batch:
                raw_stripped = raw.strip()
                if not raw_stripped:
                    continue

                # Parse JSONL record
                try:
                    rec = json.loads(raw_stripped)
                except Exception as e:
                    print(f"line={global_ln} | level=? | NO | error=malformed_json ({e})")
                    batch_ok = False
                    continue

                input_text = rec.get("input", "")
                if "create_next_map" not in input_text:
                    # Not a map line
                    continue

                output_val = rec.get("output")
                if output_val is None:
                    print(f"line={global_ln} | level=? | NO | error=missing_output")
                    batch_ok = False
                    continue

                try:
                    result = validate_output_json(output_val)
                except Exception as e:
                    print(f"line={global_ln} | level=? | NO | error=failed_to_parse_output ({e})")
                    batch_ok = False
                    continue

                level = result["level"]
                rc = result["reachable_count"]
                te = result["total_enemies"]

                # Enemy-count gating first
                if not result["enemy_count_ok"]:
                    print(
                        f"line={global_ln} | level={level} | NO | reachable={rc}/{te} | "
                        f"error={result['enemy_count_error']}"
                    )
                    batch_ok = False
                    continue

                # Then reachability
                if result["all_reachable"]:
                    print(f"line={global_ln} | level={level} | YES | reachable={rc}/{te}")
                else:
                    print(
                        f"line={global_ln} | level={level} | NO | reachable={rc}/{te} | "
                        f"unreachable={result['unreachable_enemies']}"
                    )
                    batch_ok = False

            # Batch summary + write if OK
            if batch_ok and len(batch) == batch_size:
                for (_, raw) in batch:
                    out_f.write(raw + "\n")
                wrote_any = True
                print(f"BATCH {b+1} lines {batch_first_line}-{batch_last_line}: KEEP (wrote {len(batch)} lines)")
            elif batch_ok and len(batch) != batch_size:
                for (_, raw) in batch:
                    out_f.write(raw + "\n")
                wrote_any = True
                print(
                    f"BATCH {b+1} lines {batch_first_line}-{batch_last_line}: "
                    f"KEEP (incomplete batch, wrote {len(batch)} lines)"
                )
            else:
                print(f"BATCH {b+1} lines {batch_first_line}-{batch_last_line}: DROP")

    if not wrote_any:
        print(f"No batches written. Output file '{out_path}' is empty.")


def main():
    parser = argparse.ArgumentParser(
        description="Validate map lines inside a JSONL and keep only 18-line batches with NO errors."
    )
    parser.add_argument("jsonl_path", help="Path to input JSONL (e.g., maps.jsonl)")
    parser.add_argument(
        "--out",
        default="noerror.jsonl",
        help="Path to write kept batches (default: noerror.jsonl)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=18,
        help="Batch size in lines (default: 18)",
    )
    args = parser.parse_args()

    lines = read_all_lines(args.jsonl_path)
    if not lines:
        print("Input file is empty.")
        sys.exit(0)

    process_batches(lines, args.out, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
