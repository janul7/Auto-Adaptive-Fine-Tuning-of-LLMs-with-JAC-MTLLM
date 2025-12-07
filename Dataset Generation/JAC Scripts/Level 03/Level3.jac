import json;
import random;


walker Stage1V1 {
    has current_level: int = 2;
    has difficulty: int = 1;

    # ---------- helpers ----------
     def fmt_level_text(level: dict) -> str {
        return "Level(name=" + str(level["name"]) + ", difficulty=" + str(level["difficulty"]) + ", time=" + str(
            level["time"]
        ) + ", width=" + str(level["width"]) + ", height=" + str(level["height"]) + ", num_wall=" + str(
            level["num_wall"]
        ) + ", num_enemies=" + str(level["num_enemies"]) + ")";
    }

    def fmt_position(x: int, y: int) -> str {
        return "Position(x=" + str(x) + ", y=" + str(y) + ")";
    }

    def fmt_wall_text(w: dict) -> str {
        let sx = w["start_pos"]["x"];
        let sy = w["start_pos"]["y"];
        let ex = w["end_pos"]["x"];
        let ey = w["end_pos"]["y"];
        return "Wall(start_pos=Position(x=" + str(sx) + ", y=" + str(sy) + "), end_pos=Position(x=" + str(
            ex
        ) + ", y=" + str(ey) + "))";
    }

    def join_csv(items: list) -> str {
        let out = "";
        let i = 0;
        while i < len(items) {
            if i > 0 {
                out += ", ";
            }
            out += items[i];
            i += 1;
        }
        return out;
    }

    def fmt_map_text(m: dict) -> str {
        let lvl_txt = self.fmt_level_text(m["level"]);
        let wtxts = [];
        for w in m["walls"] {
            wtxts.append(self.fmt_wall_text(w));
        }
        let walls_join = self.join_csv(wtxts);
        let otxts = [];
        for p in m["small_obstacles"] {
            otxts.append(self.fmt_position(p["x"], p["y"]));
        }
        let obs_join = self.join_csv(otxts);
        let etxts = [];
        for e in m["enemies"] {
            etxts.append(self.fmt_position(e["x"], e["y"]));
        }
        let enemies_join = self.join_csv(etxts);
        let pp = m["player_pos"];
        let ptxt = self.fmt_position(pp["x"], pp["y"]);
        return "Map_tiles(level=" + lvl_txt + ", walls=[" + walls_join + "], small_obstacles=[" + obs_join + "], enemies=[" + enemies_join + "], player_pos=" + ptxt + ")";
    }

    def seg_len_bounds(n: int) -> tuple {
        return (n // 2 - 1, n // 2 + 1);
    }

    def in_bounds(x: int, y: int, w: int, h: int) -> bool {
        return (x >= 0) and (x < w) and (y >= 0) and (y < h) ;
    }

    def neighbors4(x: int, y: int, w: int, h: int) -> list {
        let out = [];
        if self.in_bounds(x + 1, y, w, h) {
            out.append((x + 1, y));
        }
        if self.in_bounds(x - 1, y, w, h) {
            out.append((x - 1, y));
        }
        if self.in_bounds(x, y + 1, w, h) {
            out.append((x, y + 1));
        }
        if self.in_bounds(x, y - 1, w, h) {
            out.append((x, y - 1));
        }
        return out;
    }

    def bfs_reachable(w: int, h: int, blocked: set, start: tuple) -> set {
        let seen = set();
        let q = [];
        q.append(start);
        seen.add(str(start[0]) + "_" + str(start[1]));
        while len(q) > 0 {
            let cur = q.pop(0);
            let cx = cur[0];
            let cy = cur[1];
            for nb in self.neighbors4(cx, cy, w, h) {
                let k = str(nb[0]) + "_" + str(nb[1]);
                if k in blocked {
                    continue;
                }
                if k in seen {
                    continue;
                }
                seen.add(k);
                q.append(nb);
            }
        }
        return seen;
    }

    def all_enemies_reachable(
        w: int,
        h: int,
        blocked: set,
        player_xy: tuple,
        enemies: list
    ) -> bool {
        let seen = self.bfs_reachable(w, h, blocked, player_xy);
        let i = 0;
        while i < len(enemies) {
            let e = enemies[i];
            let ek = str(e["x"]) + "_" + str(e["y"]);
            if ek not in seen {
                return False;
            }
            i += 1;
        }
        return True;
    }

    def is_enemy_key(k: str, enemies: list) -> bool {
        let i = 0;
        while i < len(enemies) {
            let ek = str(enemies[i]["x"]) + "_" + str(enemies[i]["y"]);
            if k == ek {
                return True;
            }
            i += 1;
        }
        return False;
    }

    def blocked_without_enemies(blocked: set, enemies: list) -> set {
        let out = set();
        for b in blocked {
            if not self.is_enemy_key(b, enemies) {
                out.add(b);
            }
        }
        return out;
    }

    def wall_cells(walls: list) -> set {
        let s = set();
        for w in walls {
            let x0 = w["start_pos"]["x"];
            let y0 = w["start_pos"]["y"];
            let x1 = w["end_pos"]["x"];
            let y1 = w["end_pos"]["y"];
            if x0 == x1 {
                let ys = min(y0, y1);
                let ye = max(y0, y1);
                for y in range(ys, ye + 1) {
                    s.add(str(x0) + "_" + str(y));
                }
            } else {
                let xs = min(x0, x1);
                let xe = max(x0, x1);
                for x in range(xs, xe + 1) {
                    s.add(str(x) + "_" + str(y0));
                }
            }
        }
        return s;
    }

    def rand_free_cell(rng: any, w: int, h: int, taken: set) -> tuple {
        let tries = 0;
        while tries < 20000 {
            let x = rng.randint(0, w - 1);
            let y = rng.randint(0, h - 1);
            let k = str(x) + "_" + str(y);
            if k not in taken {
                return (x, y);
            }
            tries += 1;
        }
        return (0, 0);
    }

    # Walls are created in interior only 
     def random_interior_segment(rng: any, w: int, h: int) -> dict {
        let (Lmin, Lmax) = self.seg_len_bounds(min(w, h));
        let vertical = (rng.random() < 0.5);
        let L = rng.randint(Lmin, Lmax);
        # number of cells in segment
        let margin =
            2;
        # keep at least one-cell gap from borders
         if vertical {
            let x_lo = margin;
            let x_hi = w - 1 - margin;
            if x_hi < x_lo {
                x_lo = 1;
                x_hi = w - 2;
            }
            let x = rng.randint(x_lo, x_hi);
            let y0_lo = margin;
            let y0_hi = (h - margin) - L;
            if y0_hi < y0_lo {
                y0_lo = 1;
                y0_hi = h - L - 1;
            }
            let y0 = rng.randint(y0_lo, y0_hi);
            return
            {"start_pos" : {"x" : x , "y" : y0 } , "end_pos" : {"x" : x , "y" : y0 + (L - 1) } };
        } else {
            let y_lo = margin;
            let y_hi = h - 1 - margin;
            if y_hi < y_lo {
                y_lo = 1;
                y_hi = h - 2;
            }
            let y = rng.randint(y_lo, y_hi);
            let x0_lo = margin;
            let x0_hi = (w - margin) - L;
            if x0_hi < x0_lo {
                x0_lo = 1;
                x0_hi = w - L - 1;
            }
            let x0 = rng.randint(x0_lo, x0_hi);
            return
            {"start_pos" : {"x" : x0 , "y" : y } , "end_pos" : {"x" : x0 + (L - 1) , "y" : y } };
        }
    }

    can start with `root entry {
        system =
            "This is a task you must complete by returning only the output.\nDo not include explanations, code, or extra text—only the result.\n";
        # ---------- Stage 1 (Level 3) ----------
        stage1_input =
            "create_next_level\n\n" + "self = LevelManager(current_level=" + str(
                self.current_level
            ) + ", current_difficulty=" + str(self.difficulty) + ", prev_levels=[], prev_level_maps=[])\n" + "last_levels = []\n" + "difficulty = " + str(
                self.difficulty
            );
        level =

            {"name" : self.current_level + 1 , # 3 when current_level=2
            "difficulty" : self.difficulty , # 1
            "time" : 300 , "width" : 10 , "height" : 10 , "num_wall" : 5 , "num_enemies" : 2 };
        rec1 =

            {"system" : system , "input" : stage1_input , "output" : json.dumps(level, ensure_ascii=False) };
        # ---------- Stage 2 (Level 3 map) ----------
        let lvl_txt =
            self.fmt_level_text(level);
        stage2_input =
            "create_next_map\n\n" + "self = LevelManager(current_level=" + str(
                self.current_level
            ) + ", current_difficulty=" + str(self.difficulty) + ", prev_levels=[" + lvl_txt + "], prev_level_maps=[])\n" + "level = " + lvl_txt;
        let w = level["width"];
        let h = level["height"];
        let rng = random.Random();
        walls = [];
        while len(walls) < level["num_wall"] {
            walls.append(self.random_interior_segment(rng, w, h));
        }
        # ------- placements with reachability guarantees -------
        let occ =
            self.wall_cells(walls);
        # Player
        let pxy =
            self.rand_free_cell(rng, w, h, occ);
        let pk = str(pxy[0]) + "_" + str(pxy[1]);
        player_pos = {"x" : pxy[0] , "y" : pxy[1] };
        occ.add(pk);
        # Enemies 
        enemies =
            [];
        let tries_enemy = 0;
        while len(enemies) < level["num_enemies"] and tries_enemy < 20000  {
            let e = self.rand_free_cell(rng, w, h, occ);
            let ek = str(e[0]) + "_" + str(e[1]);
            if ek in occ {
                tries_enemy += 1;
                continue;
            }
            let temp_enemies = [];
            let j = 0;
            while j < len(enemies) {
                temp_enemies.append(enemies[j]);
                j += 1;
            }
            temp_enemies.append({"x" : e[0] , "y" : e[1] });
            # Build blocked set that excludes enemy cells
            let blocked_test =
                self.blocked_without_enemies(occ, temp_enemies);
            if self.all_enemies_reachable(w, h, blocked_test, pxy, temp_enemies) {
                occ.add(ek);
                # reserve the cell
                enemies.append(
                    {"x" : e[0] , "y" : e[1] }
                );
            }
            tries_enemy += 1;
        }
        # Small obstacles (match num_wall) BUT never break reachability
        small_obstacles =
            [];
        let target_obstacles = level["num_wall"];
        let tries_obs = 0;
        while len(small_obstacles) < target_obstacles and tries_obs < 40000  {
            let q = self.rand_free_cell(rng, w, h, occ);
            let k = str(q[0]) + "_" + str(q[1]);
            if k in occ {
                tries_obs += 1;
                continue;
            }
            let blocked_try = set();
            for b in occ {
                blocked_try.add(b);
            }
            blocked_try.add(k);
            # Enemy cells should not be considered blocked for pathing
            let blocked_try2 =
                self.blocked_without_enemies(blocked_try, enemies);
            if self.all_enemies_reachable(w, h, blocked_try2, pxy, enemies) {
                occ.add(k);
                small_obstacles.append({"x" : q[0] , "y" : q[1] });
            }
            tries_obs += 1;
        }
        map_out =

            {"level" : level , "walls" : walls , "small_obstacles" : small_obstacles , "enemies" : enemies , "player_pos" : player_pos };
        rec2 =

            {"system" : system , "input" : stage2_input , "output" : json.dumps(map_out, ensure_ascii=False) };
        # ---------- Stage 1B (Level 4 metadata) ----------
        let prev_lvl_txt =
            lvl_txt;
        let prev_map_txt = self.fmt_map_text(map_out);
        let next_current_level = level["name"];
        # 3
        let next_difficulty =
            self.difficulty + 1;
        # 2
        stage1b_input =
            "create_next_level\n\n" + "self = LevelManager(current_level=" + str(
                next_current_level
            ) + ", current_difficulty=" + str(next_difficulty) + ", prev_levels=[" + prev_lvl_txt + "], prev_level_maps=[" + prev_map_txt + "])\n" + "last_levels = [" + prev_lvl_txt + "]\n" + "difficulty = " + str(
                next_difficulty
            );
        level2 =

            {"name" : level["name"] + 1 , # 4
            "difficulty" : next_difficulty , # 2
            "time" : level["time"] + 100 , # 400
            "width" : level["width"] + 2 , # 12
            "height" : level["height"] + 2 , # 12
            "num_wall" : level["num_wall"] + 1 , # 6
            "num_enemies" : level["num_enemies"] + 1
            # 3
             };
        rec3 =

            {"system" : system , "input" : stage1b_input , "output" : json.dumps(level2, ensure_ascii=False) };
        # ---------- Stage 2B (Level 4 map) ----------
        let lvl2_txt =
            self.fmt_level_text(level2);
        stage2b_input =
            "create_next_map\n\n" + "self = LevelManager(current_level=" + str(
                next_current_level
            ) + ", current_difficulty=" + str(next_difficulty) + ", prev_levels=[" + prev_lvl_txt + ", " + lvl2_txt + "], prev_level_maps=[" + prev_map_txt + "])\n" + "level = " + lvl2_txt;
        let w2 = level2["width"];
        let h2 = level2["height"];
        walls2 = [];
        while len(walls2) < level2["num_wall"] {
            walls2.append(self.random_interior_segment(rng, w2, h2));
        }
        let occ2 = self.wall_cells(walls2);
        # Player (Level 4)
        let p2 =
            self.rand_free_cell(rng, w2, h2, occ2);
        let pk2 = str(p2[0]) + "_" + str(p2[1]);
        player_pos2 = {"x" : p2[0] , "y" : p2[1] };
        occ2.add(pk2);
        # Enemies (3) 
        enemies2 =
            [];
        let tries_enemy2 = 0;
        while len(enemies2) < level2["num_enemies"] and tries_enemy2 < 30000  {
            let e2 = self.rand_free_cell(rng, w2, h2, occ2);
            let ke2 = str(e2[0]) + "_" + str(e2[1]);
            if ke2 in occ2 {
                tries_enemy2 += 1;
                continue;
            }
            let temp_enemies2 = [];
            let j2 = 0;
            while j2 < len(enemies2) {
                temp_enemies2.append(enemies2[j2]);
                j2 += 1;
            }
            temp_enemies2.append({"x" : e2[0] , "y" : e2[1] });
            let blocked_test2 = self.blocked_without_enemies(occ2, temp_enemies2);
            if self.all_enemies_reachable(w2, h2, blocked_test2, p2, temp_enemies2) {
                occ2.add(ke2);
                enemies2.append({"x" : e2[0] , "y" : e2[1] });
            }
            tries_enemy2 += 1;
        }
        # Small obstacles 
        small_obstacles2 =
            [];
        let target_obstacles2 = level2["num_wall"];
        let tries_obs2 = 0;
        while len(small_obstacles2) < target_obstacles2 and tries_obs2 < 60000  {
            let q2 = self.rand_free_cell(rng, w2, h2, occ2);
            let k2 = str(q2[0]) + "_" + str(q2[1]);
            if k2 in occ2 {
                tries_obs2 += 1;
                continue;
            }
            let blocked_try2 = set();
            for b2 in occ2 {
                blocked_try2.add(b2);
            }
            blocked_try2.add(k2);
            let blocked_try2b = self.blocked_without_enemies(blocked_try2, enemies2);
            if self.all_enemies_reachable(w2, h2, blocked_try2b, p2, enemies2) {
                occ2.add(k2);
                small_obstacles2.append({"x" : q2[0] , "y" : q2[1] });
            }
            tries_obs2 += 1;
        }
        map_out2 =

            {"level" : level2 , "walls" : walls2 , "small_obstacles" : small_obstacles2 , "enemies" : enemies2 , "player_pos" : player_pos2 };
        rec4 =

            {"system" : system , "input" : stage2b_input , "output" : json.dumps(map_out2, ensure_ascii=False) };
        # -------- persist exactly four lines per run --------
         with open("levels.jsonl", "a", encoding="utf-8") as f  {
            f.write(json.dumps(rec1, ensure_ascii=False) + "\n");
            f.write(json.dumps(rec2, ensure_ascii=False) + "\n");
            f.write(json.dumps(rec3, ensure_ascii=False) + "\n");
            f.write(json.dumps(rec4, ensure_ascii=False) + "\n");
        }
    }
}


with entry:__main__ {
    root spawn Stage1V1();

# appends 4 JSONL records to levels.jsonl
 }
