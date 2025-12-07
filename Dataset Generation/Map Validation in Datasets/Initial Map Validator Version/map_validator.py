import json
from collections import deque

# Your output string (use triple quotes if multi-line)
output = '''{\"level\": {\"name\": 6, \"difficulty\": 4, \"time\": 600, \"width\": 16, \"height\": 16, \"num_wall\": 8, \"num_enemies\": 5}, \"walls\": [{\"start_pos\": {\"x\": 7, \"y\": 6}, \"end_pos\": {\"x\": 9, \"y\": 6}}, {\"start_pos\": {\"x\": 0, \"y\": 15}, \"end_pos\": {\"x\": 15, \"y\": 15}}, {\"start_pos\": {\"x\": 6, \"y\": 7}, \"end_pos\": {\"x\": 6, \"y\": 14}}, {\"start_pos\": {\"x\": 6, \"y\": 15}, \"end_pos\": {\"x\": 6, \"y\": 15}}, {\"start_pos\": {\"x\": 0, \"y\": 8}, \"end_pos\": {\"x\": 0, \"y\": 8}}, {\"start_pos\": {\"x\": 0, \"y\": 5}, \"end_pos\": {\"x\": 7, \"y\": 5}}, {\"start_pos\": {\"x\": 13, \"y\": 12}, \"end_pos\": {\"x\": 13, \"y\": 15}}, {\"start_pos\": {\"x\": 9, \"y\": 11}, \"end_pos\": {\"x\": 9, \"y\": 13}}], \"small_obstacles\": [{\"x\": 12, \"y\": 2}, {\"x\": 14, \"y\": 10}, {\"x\": 4, \"y\": 8}, {\"x\": 1, \"y\": 7}, {\"x\": 12, \"y\": 9}, {\"x\": 5, \"y\": 4}, {\"x\": 11, \"y\": 5}, {\"x\": 10, \"y\": 14}], \"enemies\": [{\"x\": 8, \"y\": 5}, {\"x\": 7, \"y\": 4}, {\"x\": 0, \"y\": 3}, {\"x\": 5, \"y\": 12}, {\"x\": 9, \"y\": 5}], \"player_pos\": {\"x\": 4, \"y\": 6}}'''

data = json.loads(output)

width = data["level"]["width"]
height = data["level"]["height"]

# Build blocked set
blocked = set()
for wall in data["walls"]:
    x1, y1 = wall["start_pos"]["x"], wall["start_pos"]["y"]
    x2, y2 = wall["end_pos"]["x"], wall["end_pos"]["y"]
    if x1 == x2:  # vertical wall
        for y in range(min(y1, y2), max(y1, y2)+1):
            blocked.add((x1, y))
    elif y1 == y2:  # horizontal wall (not present in this map, but good for generality)
        for x in range(min(x1, x2), max(x1, x2)+1):
            blocked.add((x, y1))
    else:
        # handle diagonal walls if any, but usually not needed
        pass

for obs in data["small_obstacles"]:
    blocked.add((obs["x"], obs["y"]))

# BFS
player = data["player_pos"]
queue = deque()
visited = set()
queue.append((player["x"], player["y"]))
visited.add((player["x"], player["y"]))
directions = [(0,1), (1,0), (0,-1), (-1,0)]

while queue:
    x, y = queue.popleft()
    for dx, dy in directions:
        nx, ny = x+dx, y+dy
        if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in blocked and (nx, ny) not in visited:
            visited.add((nx, ny))
            queue.append((nx, ny))

# Check if all enemies are reachable
all_reachable = True
for enemy in data["enemies"]:
    if (enemy["x"], enemy["y"]) not in visited:
        all_reachable = False
        break

if all_reachable:
    print("YES: The player can reach all enemies!")
else:
    print("NO: The player cannot reach all enemies.")

# For debugging, you can also print which enemies are not reachable:
for enemy in data["enemies"]:
    if (enemy["x"], enemy["y"]) not in visited:
        print(f"Enemy at ({enemy['x']}, {enemy['y']}) is NOT reachable.")
