


## Difficulty-3 Map Generators  

Lightweight **Higher-level scripts** for generating ** best quality, best difficulty-level maps** intended primarily for **dataset generation**.

---
This include python script which generate higher quality and complex map generating script which generates maps upto lv 10 u can visualize its outputs using map view or sample level images are included . All the maps generated are equally complex even for higher levels.

### **Script: `Dif_Lv3_V1.py and Dif_Lv3_V2.py`**  

- **Purpose:**  
  - Generate quick test maps.  
  - Print JSONL-ready lines for datasets and viewers.  
  - By design, a single run emits **18 lines**: Stage-1 metadata and Stage-2 randomized map.

- **How to Use:**  
  1. Open this folder in **VS Code** (or any IDE) and use the **integrated terminal**.  
  2. Run one of the following command in the terminal:  
  
     ```powershell
     python Dif_Lv3.py
     ```

     

  3. **Copy** the printed lines from the terminal into a file named, for example, `sample_maps.jsonl`.  

- **Result:**  
  - Level data is printed to the terminal in JSON format (one object per line).  
  - You can immediately load the `.jsonl` file with your **map viewer**.

---

In here V1 versions enemies are always more closer to the out wals so we have fixeed that with V2 version



