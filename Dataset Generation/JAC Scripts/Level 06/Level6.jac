import json;
import random;

walker Stage1V1 {
  has current_level: int = 2;
  has difficulty: int = 1;

  # ---------- helpers ----------
  def fmt_level_text(level: dict) -> str {
    return "Level(name=" + str(level["name"])
         + ", difficulty=" + str(level["difficulty"])
         + ", time=" + str(level["time"])
         + ", width=" + str(level["width"])
         + ", height=" + str(level["height"])
         + ", num_wall=" + str(level["num_wall"])
         + ", num_enemies=" + str(level["num_enemies"])
         + ")";
  }

  def fmt_position(x: int, y: int) -> str {
    return "Position(x=" + str(x) + ", y=" + str(y) + ")";
  }

  def fmt_wall_text(w: dict) -> str {
    let sx = w["start_pos"]["x"]; let sy = w["start_pos"]["y"];
    let ex = w["end_pos"]["x"];   let ey = w["end_pos"]["y"];
    return "Wall(start_pos=Position(x=" + str(sx) + ", y=" + str(sy)
         + "), end_pos=Position(x=" + str(ex) + ", y=" + str(ey) + "))";
  }

  def join_csv(items: list) -> str {
    let out = "";
    let i = 0;
    while i < len(items) {
      if i > 0 { out += ", "; }
      out += items[i];
      i += 1;
    }
    return out;
  }

  def fmt_map_text(m: dict) -> str {
    let lvl_txt = self.fmt_level_text(m["level"]);

    let wtxts = [];
    for w in m["walls"] { wtxts.append(self.fmt_wall_text(w)); }
    let walls_join = self.join_csv(wtxts);

    let otxts = [];
    for p in m["small_obstacles"] { otxts.append(self.fmt_position(p["x"], p["y"])); }
    let obs_join = self.join_csv(otxts);

    let etxts = [];
    for e in m["enemies"] { etxts.append(self.fmt_position(e["x"], e["y"])); }
    let enemies_join = self.join_csv(etxts);

    let pp = m["player_pos"];
    let ptxt = self.fmt_position(pp["x"], pp["y"]);

    return "Map_tiles(level=" + lvl_txt + ", walls=[" + walls_join
         + "], small_obstacles=[" + obs_join + "], enemies=[" + enemies_join
         + "], player_pos=" + ptxt + ")";
  }

  def seg_len_bounds(n: int) -> tuple {
    # Inclusive range: [n//2 - 1, n//2 + 1]
    # Examples: n=10 -> [4,6], n=12 -> [5,7]
    return (n//2 - 1, n//2 + 1);
  }

  # ----- Reachability utils -----
  def in_bounds(x: int, y: int, w: int, h: int) -> bool {
    return (x >= 0) and (x < w) and (y >= 0) and (y < h);
  }

  def neighbors4(x: int, y: int, w: int, h: int) -> list {
    let out = [];
    if self.in_bounds(x+1, y, w, h) { out.append((x+1, y)); }
    if self.in_bounds(x-1, y, w, h) { out.append((x-1, y)); }
    if self.in_bounds(x, y+1, w, h) { out.append((x, y+1)); }
    if self.in_bounds(x, y-1, w, h) { out.append((x, y-1)); }
    return out;
  }

  def bfs_reachable(w: int, h: int, blocked: set, start: tuple) -> set {
    let seen = set();
    let q = [];
    q.append(start);
    seen.add(str(start[0]) + "_" + str(start[1]));
    while len(q) > 0 {
      let cur = q.pop(0);
      let cx = cur[0]; let cy = cur[1];
      for nb in self.neighbors4(cx, cy, w, h) {
        let k = str(nb[0]) + "_" + str(nb[1]);
        if k in blocked { continue; }
        if k in seen { continue; }
        seen.add(k);
        q.append(nb);
      }
    }
    return seen;
  }

  def all_enemies_reachable(w: int, h: int, blocked: set, player_xy: tuple, enemies: list) -> bool {
    let seen = self.bfs_reachable(w, h, blocked, player_xy);
    let i = 0;
    while i < len(enemies) {
      let e = enemies[i];
      let ek = str(e["x"]) + "_" + str(e["y"]);
      if ek not in seen { return False; }
      i += 1;
    }
    return True;
  }

  # --- helpers to avoid treating enemy cells as blocked during BFS ---
  def is_enemy_key(k: str, enemies: list) -> bool {
    let i = 0;
    while i < len(enemies) {
      let ek = str(enemies[i]["x"]) + "_" + str(enemies[i]["y"]);
      if k == ek { return True; }
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
      let x0 = w["start_pos"]["x"]; let y0 = w["start_pos"]["y"];
      let x1 = w["end_pos"]["x"];   let y1 = w["end_pos"]["y"];
      if x0 == x1 {
        let ys = min(y0, y1); let ye = max(y0, y1);
        for y in range(ys, ye + 1) { s.add(str(x0) + "_" + str(y)); }
      } else {
        let xs = min(x0, x1); let xe = max(x0, x1);
        for x in range(xs, xe + 1) { s.add(str(x) + "_" + str(y0)); }
      }
    }
    return s;
  }

  def rand_free_cell(rng: any, w: int, h: int, taken: set) -> tuple {
    let tries = 0;
    while tries < 20000 {
      let x = rng.randint(0, w-1); let y = rng.randint(0, h-1);
      let k = str(x) + "_" + str(y);
      if k not in taken { return (x, y); }
      tries += 1;
    }
    return (0, 0);
  }

  # strictly interior walls
  def random_interior_segment(rng: any, w: int, h: int) -> dict {
    let (Lmin, Lmax) = self.seg_len_bounds(min(w, h));
    let vertical = (rng.random() < 0.5);
    let L = rng.randint(Lmin, Lmax);  # number of cells in segment
    let margin = 2;  # keep at least one-cell gap from borders

    if vertical {
      let x_lo = margin; let x_hi = w - 1 - margin;
      if x_hi < x_lo { x_lo = 1; x_hi = w - 2; }  # fallback
      let x = rng.randint(x_lo, x_hi);

      let y0_lo = margin; let y0_hi = (h - margin) - L;
      if y0_hi < y0_lo { y0_lo = 1; y0_hi = h - L - 1; }  # fallback
      let y0 = rng.randint(y0_lo, y0_hi);

      return {"start_pos": {"x": x, "y": y0}, "end_pos": {"x": x, "y": y0 + (L-1)}};
    } else {
      let y_lo = margin; let y_hi = h - 1 - margin;
      if y_hi < y_lo { y_lo = 1; y_hi = h - 2; }  # fallback
      let y = rng.randint(y_lo, y_hi);

      let x0_lo = margin; let x0_hi = (w - margin) - L;
      if x0_hi < x0_lo { x0_lo = 1; x0_hi = w - L - 1; }  # fallback
      let x0 = rng.randint(x0_lo, x0_hi);

      return {"start_pos": {"x": x0, "y": y}, "end_pos": {"x": x0 + (L-1), "y": y}};
    }
  }

  can start with `root entry {
    # EXACT system prompt
    system = "This is a task you must complete by returning only the output.\nDo not include explanations, code, or extra text—only the result.\n";

    # ---------- Stage 1 (Level 3) ----------
    stage1_input =
      "create_next_level\n\n"
      + "self = LevelManager(current_level=" + str(self.current_level)
      + ", current_difficulty=" + str(self.difficulty)
      + ", prev_levels=[], prev_level_maps=[])\n"
      + "last_levels = []\n"
      + "difficulty = " + str(self.difficulty);

    level = {
      "name": self.current_level + 1,  # 3 when current_level=2
      "difficulty": self.difficulty,   # 1
      "time": 300,
      "width": 10,
      "height": 10,
      "num_wall": 5,
      "num_enemies": 2
    };

    rec1 = {
      "system": system,
      "input": stage1_input,
      "output": json.dumps(level, ensure_ascii=False)
    };

    # ---------- Stage 2 (Level 3 map) ----------
    let lvl_txt = self.fmt_level_text(level);
    stage2_input =
      "create_next_map\n\n"
      + "self = LevelManager(current_level=" + str(self.current_level)
      + ", current_difficulty=" + str(self.difficulty)
      + ", prev_levels=[" + lvl_txt + "], prev_level_maps=[])\n"
      + "level = " + lvl_txt;

    let w = level["width"]; let h = level["height"];
    let rng = random.Random();  # no fixed seed -> varies per run

    walls = [];
    while len(walls) < level["num_wall"] {
      walls.append(self.random_interior_segment(rng, w, h));
    }

    # ------- placements with reachability guarantees -------
    let occ = self.wall_cells(walls);

    # Player
    let pxy = self.rand_free_cell(rng, w, h, occ);
    let pk  = str(pxy[0]) + "_" + str(pxy[1]);
    player_pos = {"x": pxy[0], "y": pxy[1]};
    occ.add(pk);

    # Enemies (ensure reachable; enemy tiles not blocked during BFS)
    enemies = [];
    let tries_enemy = 0;
    while len(enemies) < level["num_enemies"] and tries_enemy < 20000 {
      let e = self.rand_free_cell(rng, w, h, occ);
      let ek = str(e[0]) + "_" + str(e[1]);
      if ek in occ { tries_enemy += 1; continue; }

      let temp_enemies = [];
      let j = 0;
      while j < len(enemies) { temp_enemies.append(enemies[j]); j += 1; }
      temp_enemies.append({"x": e[0], "y": e[1]});

      let blocked_test = self.blocked_without_enemies(occ, temp_enemies);
      if self.all_enemies_reachable(w, h, blocked_test, pxy, temp_enemies) {
        occ.add(ek);
        enemies.append({"x": e[0], "y": e[1]});
      }
      tries_enemy += 1;
    }

    # Small obstacles (== num_wall) while preserving reachability
    small_obstacles = [];
    let target_obstacles = level["num_wall"];
    let tries_obs = 0;

    while len(small_obstacles) < target_obstacles and tries_obs < 40000 {
      let q = self.rand_free_cell(rng, w, h, occ);
      let k = str(q[0]) + "_" + str(q[1]);
      if k in occ { tries_obs += 1; continue; }

      let blocked_try = set();
      for b in occ { blocked_try.add(b); }
      blocked_try.add(k);

      let blocked_try2 = self.blocked_without_enemies(blocked_try, enemies);
      if self.all_enemies_reachable(w, h, blocked_try2, pxy, enemies) {
        occ.add(k);
        small_obstacles.append({"x": q[0], "y": q[1]});
      }
      tries_obs += 1;
    }

    map_out = {
      "level": level,
      "walls": walls,
      "small_obstacles": small_obstacles,
      "enemies": enemies,
      "player_pos": player_pos
    };

    rec2 = {
      "system": system,
      "input": stage2_input,
      "output": json.dumps(map_out, ensure_ascii=False)
    };

    # ---------- Stage 1B (Level 4) ----------
    let prev_lvl_txt = lvl_txt;
    let prev_map_txt = self.fmt_map_text(map_out);

    let next_current_level = level["name"];       # 3
    let next_difficulty    = self.difficulty + 1; # 2

    stage1b_input =
      "create_next_level\n\n"
      + "self = LevelManager(current_level=" + str(next_current_level)
      + ", current_difficulty=" + str(next_difficulty)
      + ", prev_levels=[" + prev_lvl_txt + "], prev_level_maps=[" + prev_map_txt + "])\n"
      + "last_levels = [" + prev_lvl_txt + "]\n"
      + "difficulty = " + str(next_difficulty);

    level2 = {
      "name": level["name"] + 1,               # 4
      "difficulty": next_difficulty,           # 2
      "time": level["time"] + 100,             # 400
      "width": level["width"] + 2,             # 12
      "height": level["height"] + 2,           # 12
      "num_wall": level["num_wall"] + 1,       # 6
      "num_enemies": level["num_enemies"] + 1  # 3
    };

    rec3 = {
      "system": system,
      "input": stage1b_input,
      "output": json.dumps(level2, ensure_ascii=False)
    };

    # ---------- Stage 2B (Level 4 map) ----------
    let lvl2_txt = self.fmt_level_text(level2);
    stage2b_input =
      "create_next_map\n\n"
      + "self = LevelManager(current_level=" + str(next_current_level)
      + ", current_difficulty=" + str(next_difficulty)
      + ", prev_levels=[" + prev_lvl_txt + ", " + lvl2_txt + "], prev_level_maps=[" + prev_map_txt + "])\n"
      + "level = " + lvl2_txt;

    let w2 = level2["width"]; let h2 = level2["height"];

    walls2 = [];
    while len(walls2) < level2["num_wall"] {
      walls2.append(self.random_interior_segment(rng, w2, h2));
    }

    let occ2 = self.wall_cells(walls2);

    # Player (Level 4)
    let p2 = self.rand_free_cell(rng, w2, h2, occ2);
    let pk2 = str(p2[0]) + "_" + str(p2[1]);
    player_pos2 = {"x": p2[0], "y": p2[1]};
    occ2.add(pk2);

    # Enemies (3) reachable
    enemies2 = [];
    let tries_enemy2 = 0;
    while len(enemies2) < level2["num_enemies"] and tries_enemy2 < 30000 {
      let e2 = self.rand_free_cell(rng, w2, h2, occ2);
      let ke2 = str(e2[0]) + "_" + str(e2[1]);
      if ke2 in occ2 { tries_enemy2 += 1; continue; }

      let temp_enemies2 = [];
      let j2 = 0;
      while j2 < len(enemies2) { temp_enemies2.append(enemies2[j2]); j2 += 1; }
      temp_enemies2.append({"x": e2[0], "y": e2[1]});

      let blocked_test2 = self.blocked_without_enemies(occ2, temp_enemies2);
      if self.all_enemies_reachable(w2, h2, blocked_test2, p2, temp_enemies2) {
        occ2.add(ke2);
        enemies2.append({"x": e2[0], "y": e2[1]});
      }
      tries_enemy2 += 1;
    }

    # Small obstacles (== 6) while preserving reachability
    small_obstacles2 = [];
    let target_obstacles2 = level2["num_wall"];
    let tries_obs2 = 0;

    while len(small_obstacles2) < target_obstacles2 and tries_obs2 < 60000 {
      let q2 = self.rand_free_cell(rng, w2, h2, occ2);
      let k2 = str(q2[0]) + "_" + str(q2[1]);
      if k2 in occ2 { tries_obs2 += 1; continue; }

      let blocked_try2 = set();
      for b2 in occ2 { blocked_try2.add(b2); }
      blocked_try2.add(k2);

      let blocked_try2b = self.blocked_without_enemies(blocked_try2, enemies2);
      if self.all_enemies_reachable(w2, h2, blocked_try2b, p2, enemies2) {
        occ2.add(k2);
        small_obstacles2.append({"x": q2[0], "y": q2[1]});
      }
      tries_obs2 += 1;
    }

    map_out2 = {
      "level": level2,
      "walls": walls2,
      "small_obstacles": small_obstacles2,
      "enemies": enemies2,
      "player_pos": player_pos2
    };

    rec4 = {
      "system": system,
      "input": stage2b_input,
      "output": json.dumps(map_out2, ensure_ascii=False)
    };

    # ---------- Stage 1C (Level 5) ----------
    let prev_lvl_txt2 = lvl2_txt;
    let prev_map_txt2 = self.fmt_map_text(map_out2);

    let next_current_level2 = level2["name"];       # 4
    let next_difficulty2    = next_difficulty + 1;  # 3

    stage1c_input =
      "create_next_level\n\n"
      + "self = LevelManager(current_level=" + str(next_current_level2)
      + ", current_difficulty=" + str(next_difficulty2)
      + ", prev_levels=[" + prev_lvl_txt + ", " + prev_lvl_txt2 + "], prev_level_maps=["
      + prev_map_txt + ", " + prev_map_txt2 + "])\n"
      + "last_levels = [" + prev_lvl_txt + ", " + prev_lvl_txt2 + "]\n"
      + "difficulty = " + str(next_difficulty2);

    level3 = {
      "name": level2["name"] + 1,               # 5
      "difficulty": next_difficulty2,           # 3
      "time": level2["time"] + 100,             # 500
      "width": level2["width"] + 2,             # 14
      "height": level2["height"] + 2,           # 14
      "num_wall": level2["num_wall"] + 1,       # 7
      "num_enemies": level2["num_enemies"] + 1  # 4
    };

    rec5 = {
      "system": system,
      "input": stage1c_input,
      "output": json.dumps(level3, ensure_ascii=False)
    };

    # ---------- Stage 2C (Level 5 map) ----------
    let lvl3_txt = self.fmt_level_text(level3);
    stage2c_input =
      "create_next_map\n\n"
      + "self = LevelManager(current_level=" + str(next_current_level2)
      + ", current_difficulty=" + str(next_difficulty)
      + ", prev_levels=[" + prev_lvl_txt + ", " + prev_lvl_txt2 + ", " + lvl3_txt + "], prev_level_maps=["
      + prev_map_txt + ", " + prev_map_txt2 + "])\n"
      + "level = " + lvl3_txt;

    let w3 = level3["width"]; let h3 = level3["height"];

    walls3 = [];
    while len(walls3) < level3["num_wall"] {
      walls3.append(self.random_interior_segment(rng, w3, h3));
    }

    let occ3 = self.wall_cells(walls3);

    # Player (Level 5)
    let p3 = self.rand_free_cell(rng, w3, h3, occ3);
    let pk3 = str(p3[0]) + "_" + str(p3[1]);
    player_pos3 = {"x": p3[0], "y": p3[1]};
    occ3.add(pk3);

    # Enemies (4) reachable
    enemies3 = [];
    let tries_enemy3 = 0;
    while len(enemies3) < level3["num_enemies"] and tries_enemy3 < 90000 {
      let e3 = self.rand_free_cell(rng, w3, h3, occ3);
      let ke3 = str(e3[0]) + "_" + str(e3[1]);
      if ke3 in occ3 { tries_enemy3 += 1; continue; }

      let temp_enemies3 = [];
      let jj = 0;
      while jj < len(enemies3) { temp_enemies3.append(enemies3[jj]); jj += 1; }
      temp_enemies3.append({"x": e3[0], "y": e3[1]});

      let blocked_test3 = self.blocked_without_enemies(occ3, temp_enemies3);
      if self.all_enemies_reachable(w3, h3, blocked_test3, p3, temp_enemies3) {
        occ3.add(ke3);
        enemies3.append({"x": e3[0], "y": e3[1]});
      }
      tries_enemy3 += 1;
    }

    # Small obstacles (== 7) preserving reachability
    small_obstacles3 = [];
    let target_obstacles3 = level3["num_wall"];
    let tries_obs3 = 0;

    while len(small_obstacles3) < target_obstacles3 and tries_obs3 < 120000 {
      let q3 = self.rand_free_cell(rng, w3, h3, occ3);
      let k3 = str(q3[0]) + "_" + str(q3[1]);
      if k3 in occ3 { tries_obs3 += 1; continue; }

      let blocked_try3 = set();
      for b3 in occ3 { blocked_try3.add(b3); }
      blocked_try3.add(k3);

      let blocked_try3b = self.blocked_without_enemies(blocked_try3, enemies3);
      if self.all_enemies_reachable(w3, h3, blocked_try3b, p3, enemies3) {
        occ3.add(k3);
        small_obstacles3.append({"x": q3[0], "y": q3[1]});
      }
      tries_obs3 += 1;
    }

    map_out3 = {
      "level": level3,
      "walls": walls3,
      "small_obstacles": small_obstacles3,
      "enemies": enemies3,
      "player_pos": player_pos3
    };

    rec6 = {
      "system": system,
      "input": stage2c_input,
      "output": json.dumps(map_out3, ensure_ascii=False)
    };

    # ---------- Stage 1D (Level 6 ) ----------
    let prev_lvl_txt3 = lvl3_txt;
    let prev_map_txt3 = self.fmt_map_text(map_out3);

    let next_current_level3 = level3["name"];     # 5
    let next_difficulty3    = next_difficulty2 + 1;  # 4

    stage1d_input =
      "create_next_level\n\n"
      + "self = LevelManager(current_level=" + str(next_current_level3)
      + ", current_difficulty=" + str(next_difficulty3)
      + ", prev_levels=[" + prev_lvl_txt + ", " + prev_lvl_txt2 + ", " + prev_lvl_txt3 + "], prev_level_maps=["
      + prev_map_txt + ", " + prev_map_txt2 + ", " + prev_map_txt3 + "])\n"
      + "last_levels = [" + prev_lvl_txt + ", " + prev_lvl_txt2 + ", " + prev_lvl_txt3 + "]\n"
      + "difficulty = " + str(next_difficulty3);

    level4 = {
      "name": level3["name"] + 1,               # 6
      "difficulty": next_difficulty3,           # 4
      "time": level3["time"] + 100,             # 600
      "width": level3["width"] + 2,             # 16
      "height": level3["height"] + 2,           # 16
      "num_wall": level3["num_wall"] + 1,       # 8
      "num_enemies": level3["num_enemies"] + 1  # 5
    };

    rec7 = {
      "system": system,
      "input": stage1d_input,
      "output": json.dumps(level4, ensure_ascii=False)
    };

    # ---------- Stage 2D (Level 6 map) ----------
    let lvl4_txt = self.fmt_level_text(level4);
    stage2d_input =
      "create_next_map\n\n"
      + "self = LevelManager(current_level=" + str(next_current_level3)
      + ", current_difficulty=" + str(next_difficulty3 - 1)
      + ", prev_levels=[" + prev_lvl_txt + ", " + prev_lvl_txt2 + ", " + prev_lvl_txt3 + ", " + lvl4_txt + "], prev_level_maps=["
      + prev_map_txt + ", " + prev_map_txt2 + ", " + prev_map_txt3 + "])\n"
      + "level = " + lvl4_txt;

    let w4 = level4["width"]; let h4 = level4["height"];

    walls4 = [];
    while len(walls4) < level4["num_wall"] {
      walls4.append(self.random_interior_segment(rng, w4, h4));
    }

    let occ4 = self.wall_cells(walls4);

    # Player (Level 6)
    let p4 = self.rand_free_cell(rng, w4, h4, occ4);
    let pk4 = str(p4[0]) + "_" + str(p4[1]);
    player_pos4 = {"x": p4[0], "y": p4[1]};
    occ4.add(pk4);

    # Enemies (5) reachable
    enemies4 = [];
    let tries_enemy4 = 0;
    while len(enemies4) < level4["num_enemies"] and tries_enemy4 < 120000 {
      let e4 = self.rand_free_cell(rng, w4, h4, occ4);
      let ke4 = str(e4[0]) + "_" + str(e4[1]);
      if ke4 in occ4 { tries_enemy4 += 1; continue; }

      let temp_enemies4 = [];
      let j4 = 0;
      while j4 < len(enemies4) { temp_enemies4.append(enemies4[j4]); j4 += 1; }
      temp_enemies4.append({"x": e4[0], "y": e4[1]});

      let blocked_test4 = self.blocked_without_enemies(occ4, temp_enemies4);
      if self.all_enemies_reachable(w4, h4, blocked_test4, p4, temp_enemies4) {
        occ4.add(ke4);
        enemies4.append({"x": e4[0], "y": e4[1]});
      }
      tries_enemy4 += 1;
    }

    # Small obstacles (== 8) reachability-safe
    small_obstacles4 = [];
    let target_obstacles4 = level4["num_wall"];
    let tries_obs4 = 0;

    while len(small_obstacles4) < target_obstacles4 and tries_obs4 < 160000 {
      let q4 = self.rand_free_cell(rng, w4, h4, occ4);
      let k4 = str(q4[0]) + "_" + str(q4[1]);
      if k4 in occ4 { tries_obs4 += 1; continue; }

      let blocked_try4 = set();
      for b4 in occ4 { blocked_try4.add(b4); }
      blocked_try4.add(k4);

      let blocked_try4b = self.blocked_without_enemies(blocked_try4, enemies4);
      if self.all_enemies_reachable(w4, h4, blocked_try4b, p4, enemies4) {
        occ4.add(k4);
        small_obstacles4.append({"x": q4[0], "y": q4[1]});
      }
      tries_obs4 += 1;
    }

    map_out4 = {
      "level": level4,
      "walls": walls4,
      "small_obstacles": small_obstacles4,
      "enemies": enemies4,
      "player_pos": player_pos4
    };

    rec8 = {
      "system": system,
      "input": stage2d_input,
      "output": json.dumps(map_out4, ensure_ascii=False)
    };

    # ---------- Stage 1E (Level 7 metadata) ----------
    let prev_lvl_txt4 = lvl4_txt;
    let prev_map_txt4 = self.fmt_map_text(map_out4);

    let next_current_level4 = level4["name"];     # 6
    let next_difficulty4    = next_difficulty3 + 1;  # 5

    stage1e_input =
      "create_next_level\n\n"
      + "self = LevelManager(current_level=" + str(next_current_level4)
      + ", current_difficulty=" + str(next_difficulty4)
      + ", prev_levels=[" + prev_lvl_txt + ", " + prev_lvl_txt2 + ", " + prev_lvl_txt3 + ", " + prev_lvl_txt4 + "], prev_level_maps=["
      + prev_map_txt + ", " + prev_map_txt2 + ", " + prev_map_txt3 + ", " + prev_map_txt4 + "])\n"
      + "last_levels = [" + prev_lvl_txt + ", " + prev_lvl_txt2 + ", " + prev_lvl_txt3 + ", " + prev_lvl_txt4 + "]\n"
      + "difficulty = " + str(next_difficulty4);

    level5 = {
      "name": level4["name"] + 1,               # 7
      "difficulty": next_difficulty4,           # 5
      "time": level4["time"] + 100,             # 700
      "width": level4["width"] + 2,             # 18
      "height": level4["height"] + 2,           # 18
      "num_wall": level4["num_wall"] + 1,       # 9
      "num_enemies": level4["num_enemies"] + 1  # 6
    };

    rec9 = {
      "system": system,
      "input": stage1e_input,
      "output": json.dumps(level5, ensure_ascii=False)
    };

    # ---------- Stage 2E (Level 7 map) ----------
    let lvl5_txt = self.fmt_level_text(level5);
    stage2e_input =
      "create_next_map\n\n"
      + "self = LevelManager(current_level=" + str(next_current_level4)
      + ", current_difficulty=" + str(next_difficulty4 - 1)
      + ", prev_levels=[" + prev_lvl_txt + ", " + prev_lvl_txt2 + ", " + prev_lvl_txt3 + ", " + prev_lvl_txt4 + ", " + lvl5_txt + "], prev_level_maps=["
      + prev_map_txt + ", " + prev_map_txt2 + ", " + prev_map_txt3 + ", " + prev_map_txt4 + "])\n"
      + "level = " + lvl5_txt;

    let w5 = level5["width"]; let h5 = level5["height"];

    walls5 = [];
    while len(walls5) < level5["num_wall"] {
      walls5.append(self.random_interior_segment(rng, w5, h5));
    }

    let occ5 = self.wall_cells(walls5);

    # Player (Level 7)
    let p5 = self.rand_free_cell(rng, w5, h5, occ5);
    let pk5 = str(p5[0]) + "_" + str(p5[1]);
    player_pos5 = {"x": p5[0], "y": p5[1]};
    occ5.add(pk5);

    # Enemies (6) reachable
    enemies5 = [];
    let tries_enemy5 = 0;
    while len(enemies5) < level5["num_enemies"] and tries_enemy5 < 200000 {
      let e5 = self.rand_free_cell(rng, w5, h5, occ5);
      let ke5 = str(e5[0]) + "_" + str(e5[1]);
      if ke5 in occ5 { tries_enemy5 += 1; continue; }

      let temp_enemies5 = [];
      let j5 = 0;
      while j5 < len(enemies5) { temp_enemies5.append(enemies5[j5]); j5 += 1; }
      temp_enemies5.append({"x": e5[0], "y": e5[1]});

      let blocked_test5 = self.blocked_without_enemies(occ5, temp_enemies5);
      if self.all_enemies_reachable(w5, h5, blocked_test5, p5, temp_enemies5) {
        occ5.add(ke5);
        enemies5.append({"x": e5[0], "y": e5[1]});
      }
      tries_enemy5 += 1;
    }

    # Small obstacles (== 9) reachability-safe
    small_obstacles5 = [];
    let target_obstacles5 = level5["num_wall"];
    let tries_obs5 = 0;

    while len(small_obstacles5) < target_obstacles5 and tries_obs5 < 220000 {
      let q5 = self.rand_free_cell(rng, w5, h5, occ5);
      let k5 = str(q5[0]) + "_" + str(q5[1]);
      if k5 in occ5 { tries_obs5 += 1; continue; }

      let blocked_try5 = set();
      for b5 in occ5 { blocked_try5.add(b5); }
      blocked_try5.add(k5);

      let blocked_try5b = self.blocked_without_enemies(blocked_try5, enemies5);
      if self.all_enemies_reachable(w5, h5, blocked_try5b, p5, enemies5) {
        occ5.add(k5);
        small_obstacles5.append({"x": q5[0], "y": q5[1]});
      }
      tries_obs5 += 1;
    }

    map_out5 = {
      "level": level5,
      "walls": walls5,
      "small_obstacles": small_obstacles5,
      "enemies": enemies5,
      "player_pos": player_pos5
    };

    rec10 = {
      "system": system,
      "input": stage2e_input,
      "output": json.dumps(map_out5, ensure_ascii=False)
    };

    # 
    with open("levels.jsonl", "a", encoding="utf-8") as f {
      f.write(json.dumps(rec1, ensure_ascii=False) + "\n");   # L3 meta
      f.write(json.dumps(rec2, ensure_ascii=False) + "\n");   # L3 map
      f.write(json.dumps(rec3, ensure_ascii=False) + "\n");   # L4 meta
      f.write(json.dumps(rec4, ensure_ascii=False) + "\n");   # L4 map
      f.write(json.dumps(rec5, ensure_ascii=False) + "\n");   # L5 meta
      f.write(json.dumps(rec6, ensure_ascii=False) + "\n");   # L5 map
      f.write(json.dumps(rec7, ensure_ascii=False) + "\n");   # L6 meta
      f.write(json.dumps(rec8, ensure_ascii=False) + "\n");   # L6 map
      f.write(json.dumps(rec9, ensure_ascii=False) + "\n");   # L7 meta
      f.write(json.dumps(rec10, ensure_ascii=False) + "\n");  # L7 map
    }
  }
}

with entry:__main__ {
  root spawn Stage1V1();  # appends 10 JSONL records to levels.jsonl
}

