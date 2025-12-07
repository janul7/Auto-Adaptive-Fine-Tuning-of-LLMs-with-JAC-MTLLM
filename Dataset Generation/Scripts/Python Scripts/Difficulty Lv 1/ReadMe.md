## Difficulty-1 Map Generators  

Lightweight **initial-level scripts** for generating **simple, low-level maps** intended primarily for **testing**.

---

### **Script: `Dif_Lv1.py`**  

- **Purpose:**  
  - Generate quick test maps.  
  - Print JSONL-ready lines for datasets and viewers.  
  - By design, a single run emits **two lines**: Stage-1 metadata and Stage-2 randomized map.

- **How to Use:**  
  1. Open this folder in **VS Code** (or any IDE) and use the **integrated terminal**.  
  2. Run one of the following command in the terminal:  
  
     ```powershell
     python Dif_Lv1.py
     ```

     

  3. **Copy** the printed lines from the terminal into a file named, for example, `sample_maps.jsonl`.  

- **Result:**  
  - Level data is printed to the terminal in JSON format (one object per line).  
  - You can immediately load the `.jsonl` file with your **map viewer**.

---

### **Output Format (example)**  

```text
{"stage":1,"...":"..."}
{"stage":2,"...":"..."}



