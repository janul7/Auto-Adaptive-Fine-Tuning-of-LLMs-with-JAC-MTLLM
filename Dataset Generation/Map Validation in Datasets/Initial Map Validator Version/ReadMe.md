## Map Validator — Initial Version (Hardcoded Input)

Validate a **single map output** by hardcoding it into the script and running the validator.  
This mode is meant for quick checks while developing—no file I/O required.

---

### 🔧 What this does
- Reads the JSON you place in the `output` string.
- Validates the structure/rules of the map.
- Prints **PASS** if valid, otherwise **FAIL** (with details in the terminal).

---

### 🚀 How to Run (Hardcoded Mode)

1. Open `map_validator.py`.
2. Set the `output` variable to the map JSON you want to check (as a triple-quoted string):

   ```python
   # Replace "..." with the rest of your JSON
   output = '''{\"level\": {\"name\": 6, \"difficulty\": 4, \"time\": 600, ...}, 
               \"walls\": [...],
               \"small_obstacles\": [...],
               \"enemies\": [...],
               \"player_pos\": {\"x\": 4, \"y\": 6}}'''

3. Save the file.
4. Open map_validator.py in integrated terminal and run.

   ```python
   python map_validator.py

5. It will display the result in the terminal
      
