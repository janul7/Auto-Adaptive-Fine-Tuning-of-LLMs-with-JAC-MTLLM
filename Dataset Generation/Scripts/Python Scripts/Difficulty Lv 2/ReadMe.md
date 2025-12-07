
## Difficulty-2 Map Generators  

Lightweight **mdoderare-level scripts** for generating **quality, moderate-level maps** intended primarily for **dataset generation**.

---
This Includes a python script which can generate a moderate quality maps and it is effective for first levels and is it not that much effective for last levels ( last levels are less complex )

### **Script: `Dif_Lv2.py`**  

- **Purpose:**  
  - Generate quick test maps.  
  - Print JSONL-ready lines for datasets and viewers.  
  - By design, a single run emits **ten**: Stage-1 metadata and Stage-2 randomized map.

- **How to Use:**  
  1. Open this folder in **VS Code** (or any IDE) and use the **integrated terminal**.  
  2. Run one of the following command in the terminal:  
  
     ```powershell
     python Dif_Lv2.py
     ```

     

  3. **Copy** the printed lines from the terminal into a file named, for example, `sample_maps.jsonl`.  

- **Result:**  
  - Level data is printed to the terminal in JSON format (one object per line).  
  - You can immediately load the `.jsonl` file with your **map viewer**.

---





