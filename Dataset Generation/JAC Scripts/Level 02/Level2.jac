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

    def bounds_segments(w: int, h: int) -> list {
        return
        [{"start_pos" : {"x" : 0 , "y" : 0 } , "end_pos" : {"x" : w - 1 , "y" : 0 } },

        {"start_pos" : {"x" : w - 1 , "y" : 0 } , "end_pos" : {"x" : w - 1 , "y" : h - 1 } },

        {"start_pos" : {"x" : 0 , "y" : h - 1 } , "end_pos" : {"x" : w - 1 , "y" : h - 1 } },
        {"start_pos" : {"x" : 0 , "y" : 0 } , "end_pos" : {"x" : 0 , "y" : h - 1 } }];
    }

    def seg_len_bounds(n: int) -> tuple {
        return (n // 2 - 1, n // 2 + 1);
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
        while tries < 500 {
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

    def random_interior_segment(rng: any, w: int, h: int) -> dict {
        let (Lmin, Lmax) = self.seg_len_bounds(min(w, h));
        let vertical = (rng.random() < 0.5);
        let L = rng.randint(Lmin, Lmax);
        let margin = 2;
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
        # ---------- Stage 1 ----------
        stage1_input =
            "create_next_level\n\n" + "self = LevelManager(current_level=" + str(
                self.current_level
            ) + ", current_difficulty=" + str(self.difficulty) + ", prev_levels=[], prev_level_maps=[])\n" + "last_levels = []\n" + "difficulty = " + str(
                self.difficulty
            );
        level =

            {"name" : self.current_level + 1 , "difficulty" : self.difficulty , "time" : 300 , "width" : 10 , "height" : 10 , "num_wall" : 5 , "num_enemies" : 2 };
        rec1 =

            {"system" : system , "input" : stage1_input , "output" : json.dumps(level, ensure_ascii=False) };
        # ---------- Stage 2 ----------
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
        while len(walls) < 5 {
            walls.append(self.random_interior_segment(rng, w, h));
        }
        let occ = self.wall_cells(walls);
        let pxy = self.rand_free_cell(rng, w, h, occ);
        let pk = str(pxy[0]) + "_" + str(pxy[1]);
        player_pos = {"x" : pxy[0] , "y" : pxy[1] };
        occ.add(pk);
        # Small obstacles 
        small_obstacles =
            [];
        while len(small_obstacles) < 5 {
            let q = self.rand_free_cell(rng, w, h, occ);
            let k = str(q[0]) + "_" + str(q[1]);
            if k in occ {
                continue;
            }
            occ.add(k);
            small_obstacles.append({"x" : q[0] , "y" : q[1] });
        }
        # Enemies 
        enemies =
            [];
        while len(enemies) < level["num_enemies"] {
            let e = self.rand_free_cell(rng, w, h, occ);
            let k = str(e[0]) + "_" + str(e[1]);
            if k in occ {
                continue;
            }
            occ.add(k);
            enemies.append({"x" : e[0] , "y" : e[1] });
        }
        map_out =

            {"level" : level , "walls" : walls , "small_obstacles" : small_obstacles , "enemies" : enemies , "player_pos" : player_pos };
        rec2 =

            {"system" : system , "input" : stage2_input , "output" : json.dumps(map_out, ensure_ascii=False) };
        with open("levels.jsonl", "a", encoding="utf-8") as f  {
            f.write(json.dumps(rec1, ensure_ascii=False) + "\n");
            f.write(json.dumps(rec2, ensure_ascii=False) + "\n");
        }
    }
}


with entry:__main__ {
    root spawn Stage1V1();
}
