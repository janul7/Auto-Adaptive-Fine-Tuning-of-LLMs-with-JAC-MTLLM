## Map Viewer Tools  

We implemented two versions of the map viewing tool to ensure both visualization and quality validation of generated maps.  

---

### **Version 1: `map_view_v1.py`**  

- **Purpose:**  
  - View maps from a `.jsonl` file.  
  - Automatically save map visualizations as `.png` images.  

- **How to Use:**  
  1. Open the folder in VS Code that contains `sample_maps.jsonl` and `map_view_v1.py`.  
  2. No need to create an output folder; it will be generated automatically.  
  3. Run the following command in the integrated terminal:  
     ```bash
     python .\map_view_v1.py .\sample_maps.jsonl --outdir .\photos_v1 --view
     ```  

- **Result:**  
  - Displays all maps from `sample_maps.jsonl`.  
  - Saves corresponding `.png` images into the `photos_v1` folder.  

---

### **version 2: `map_view_v2.py`**  

- **Purpose (Upgraded):**  
  - View maps from a `.jsonl` file.  
  - Identify issues such as:  
    - Overlapping of player, enemy, and walls.  
    - Distinguish walls from small obstacles.  
  - Save map visualizations automatically.  

- **How to Use:**  
  1. Open the folder in VS Code that contains `sample_maps.jsonl` and `map_view_v2.py`.  
  2. Run the following command in the terminal:  
     ```bash
     python .\map_view_v2.py .\sample_maps.jsonl --outdir .\photos_v2 --view
     ```  

- **Result:**  
  - Displays all maps from `sample_maps.jsonl`.  
  - Detects overlaps and obstacles.  
  - Saves `.png` images into the `photos_v2` folder.  

---

### **Comparison: V1 vs V2**  

| Feature                              | **V1: map_view_v1.py**        | **V2: map_view_v2.py**              |
|--------------------------------------|--------------------------------|--------------------------------------|
| View maps from `.jsonl`              | ✅                             | ✅                                   |
| Save `.png` images                   | ✅ (saved in `photos_v1`)      | ✅ (saved in `photos_v2`)            |
| Automatic folder creation            | ✅                             | ✅                                   |
| Detect overlapping (player/enemy/walls) | ❌                             | ✅                                   |
| Distinguish walls vs small obstacles | ❌                             | ✅                                   |
