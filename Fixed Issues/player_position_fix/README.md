# Spud-nik : SOLO  

A simple Pygame-based top-down game where the player fights enemies on a generated map.

## What We Fixed  

- **Player Centering Bug**  
  Previously, the player was not always visible (sometimes spawning at the bottom of the map and going off-screen).  
  ✅ Fixed by adding a **camera system** in `Game.draw` to always center the player on screen.

- **Unwinnable Map Issue**  
  In some maps, the player spawned in unreachable areas where enemies couldn’t be killed.  
  ✅ Fixed by updating `game.impl.jac` to stop the game immediately when the player is not visible or reachable.

## Changes Made  

- Replaced the old draw logic with a **centered-camera draw**:
  ```python
  cam_x = (WIN_WIDTH // 2) - self.player.rect.centerx
  cam_y = (WIN_HEIGHT // 2) - self.player.rect.centery

<img width="857" height="669" alt="Screenshot 2025-09-03 044720" src="https://github.com/user-attachments/assets/c8922370-d36e-48a8-8888-496b2be88c4e" />
