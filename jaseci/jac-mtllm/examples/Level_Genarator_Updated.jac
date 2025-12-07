import from mtllm { Model }

glob llm = Model(model_name="gpt-4o-mini", verbose=True);

obj Position {
    has x: int, y: int;
}

obj Wall {
    has start_pos: Position, end_pos: Position;
}

obj Map_tiles {
    has level: Level;
    has walls: list[Wall],
        small_obstacles: list[Position];
    has enemies: list[Position];
    has player_pos: Position;
}

obj Level {
    has name: int,
        difficulty: int,
        time: int;
    has width: int,
        height: int,
        num_wall: int,
        num_enemies: int;
}

obj LevelManager {
    has current_level: int = 0, current_difficulty: int = 1,
        prev_levels: list[Level] = [], prev_level_maps: list[Map_tiles] = [];

    
    def create_next_level(last_levels: list[Level], difficulty: int)
        -> Level by llm(temperature=0.8);


    
    def create_next_map(level: Level) -> Map_tiles by llm(temperature=0.8);


    def get_next_level -> tuple(Level, Map_tiles) {
        self.current_level += 1;
        # Keeping Only the Last 3 Levels
        if len(self.prev_levels) > 3 {
            self.prev_levels.pop(0);
            self.prev_level_maps.pop(0);
        }
        # Generating the New Level
        new_level = self.create_next_level(
            self.prev_levels,
            self.current_difficulty,
        );    
        
        

        self.prev_levels.append(new_level);
        # Generating the Map of the New Level
        new_level_map = self.create_next_map(new_level);
        self.prev_level_maps.append(new_level_map);
        # Increasing the Difficulty for end of every 2 Levels
        if self.current_level % 2 == 0 {
            self.current_difficulty += 1;
        }
        return (new_level, new_level_map);
    }
}

'''Get the map of the level'''
def get_map(map: Map) -> str {
    map_tiles:list[list[str]] = [['.' for _ in range(map.level.width)] for _ in range(map.level.height)];

    for wall in map.walls {
        for x in range(wall.start_pos.x, wall.end_pos.x + 1) {
            for y in range(wall.start_pos.y, wall.end_pos.y + 1) {
                if 0 <= x < map.level.width and 0 <= y < map.level.height {
                    map_tiles[y][x] = 'B';
                }
            }
        }
    }

    for obs in map.small_obstacles {
        if 0 <= obs.x < map.level.width and 0 <= obs.y < map.level.height {
            map_tiles[obs.y][obs.x] = 'B';
        }
    }

    for enemy in map.enemies {
        if 0 <= enemy.x < map.level.width and 0 <= enemy.y < map.level.height {
            map_tiles[enemy.y][enemy.x] = 'E';
        }
    }
    if 0 <= map.player_pos.x < map.level.width and 0 <= map.player_pos.y < map.level.height {
        map_tiles[map.player_pos.y][map.player_pos.x] = 'P';
    }
    map_tiles:list[list[str]] = [['B'] + row + ['B'] for row in map_tiles];
    map_tiles:list[list[str]] = [['B' for _ in range(map.level.width + 2)]] + map_tiles + [['B' for _ in range(map.level.width + 2)]];
    return [''.join(row) for row in map_tiles];
}

with entry {
    level_manager = LevelManager();
    for i in range(8) {
        (new_level, new_level_map) = level_manager.get_next_level();
        print(new_level);
        print('\n'.join(get_map(new_level_map)));
    }
}