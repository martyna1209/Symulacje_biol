import numpy as np
import imageio
import matplotlib.pyplot as plt

# Słownik wzorów
PATTERNS = {
    # Ciągłe życie
    "block": np.array([[1,1],[1,1]], dtype=int),
    "beehive": np.array([[0,1,1,0],
                         [1,0,0,1],
                         [0,1,1,0]], dtype=int),
    "loaf": np.array([[0,1,1,0],
                      [1,0,0,1],
                      [0,1,0,1],
                      [0,0,1,0]], dtype=int),

    # Oscylatory
    "blinker": np.array([[1,1,1]], dtype=int),

    "beacon": np.array([[1,1,0,0],
                        [1,0,0,0],
                        [0,0,0,1],
                        [0,0,1,1]], dtype=int),

    # Bomba (glidery) 
    "r_pentomino": np.array([[0, 1, 1],
                             [1, 1, 0],
                             [0, 1, 0]], dtype=int),

    # Metuzalech diehard 
    "diehard": np.array([
        [0,0,0,0,0,0,1,0],
        [1,1,0,0,0,0,0,0],
        [0,1,0,0,0,1,1,1]
    ], dtype=int)
}

def place_pattern(grid, pattern_name, top, left):
    """
    Wzorzec PATTERNS umieszczamy do macierzy grid w miejscu (top, left).
    Jeśli wzorzec wykracza poza brzegi, zostaje przycięty.
    """
    if pattern_name not in PATTERNS:
        raise ValueError(f"Nie ma wzorca '{pattern_name}' w słowniku")
    pattern = PATTERNS[pattern_name]
    ph, pw = pattern.shape # Wysokość i szerokość wzorca
    H, W = grid.shape # Wymiary docelowej siatki
    if top < 0 or left < 0 or top >= H or left >= W:
        raise ValueError("Wzorzec poza granicami")
    crop_h = min(ph, H-top) # Ile wierszy zmieści się w grid od top do końca
    crop_w = min(pw, W-left)
    cropped = pattern[:crop_h, :crop_w]
    grid[top: top + crop_h, left: left + crop_w] = cropped # Jedynki w odpowiednich miejscach, zgodnie ze wzorcem

def game_of_life_step(current_grid):
    """
    Jedna epoka symulacji Conwaya.
    Oblicza liczbę sąsiadów przez przesunięcie tablicy za pomocą np.roll.
    Tworzy boolean i zwraca nową tablicę new_grid.
    - current_grid: 2D tablica NumPy typu int (0 = martwa komórka, 1 = żywa komórka).
    Zwraca nową tablicę new_grid tego samego rozmiaru, utworzoną synchronicznie według reguł:
      * Żywa komórka (1) z <2 lub >3 sąsiadami umiera,
      * Żywa komórka z 2 lub 3 sąsiadami pozostaje żywa,
      * Martwa komórka (0) z dokładnie 3 sąsiadami staje się żywa
    Warunki brzegowe są periodyczne.
    """
    # Sąsiedztwo Moore'a i warunki periodyczne
    neighbors = np.zeros_like(current_grid, dtype=int) # Tablica z zerami, rozmiar current_grid do zliczenia żywych sąsiadów
    # Sumujemy 8 sąsiednich pól dla każdej komórki, np.roll do przesunięcia tablicy o 1 w danym kierunku, dwie pętle przez kombinacje przesunięć 
    for i in [-1, 0, 1]: # Przesunięcie w wierszach góra/dół/brak w pionie 
        for j in [-1, 0, 1]: # Przesunięcie w kolumnach lewo/prawo/brak w poziomie
            if i == 0 and j == 0: 
                continue # Pomijanie (0,0)
            # Najpierw przesunięcie w pionie (axis=0), potem w poziomie (axis=1)
            # Neighbors[y,x] będzie sumą wszystkich żywych komórek, które przez przesunięcie trafiły na tą pozycję
            neighbors += np.roll(np.roll(current_grid, shift=i, axis=0), shift=j, axis=1)
    # Warunki przeżycia i narodzin, survive_mask[y,x] == True (ma 2 lub 3 sąsiadów)
    survive_mask = (current_grid == 1) & ((neighbors == 2) | (neighbors == 3))
    # Jest martwa i ma dokładnie 3 sąsiadów  i ożyje
    birth_mask = (current_grid == 0) & (neighbors == 3)
    new_grid = np.zeros_like(current_grid, dtype=int)  # Nowa tablica stanu, zera, domyślnie martwe komórki
    new_grid[survive_mask | birth_mask] = 1 # Zmiana na 1 (żywe) spełniające warunek przetrwania lub narodzin
    return new_grid # New_grid = stan gry w następnej epoce



def save_grid_scaled(grid_array, filename, scale=10):
    """
    Zapis grid_array jako obrazek PNG z paletą kolorów (czarny=0, biały=1) i wyższą rozdzielczością scale = 10.
    Uzyskujemy plik, który potem wykorzystamy do złożenia GIF.
    """

    big = np.kron(grid_array, np.ones((scale, scale), dtype=np.uint8))
    img = (big * 255).astype(np.uint8)
    imageio.imwrite(filename, img)
  

def simulate_three_still_life():
    """
    Symuluje trzy wzorce still life (block, beehive, loaf) w siatce 50×50.
    - Wklejamy każdy wzorzec, żeby się nie stykały.
    - Uruchamiamy 5 epok i zbieramy klatki.
    - Tworzymy GIF:'three_still_life.gif'.
    """
    N = 50
    epochs = 5
    grid = np.zeros((N, N), dtype=int) # Pusta plansza (wszystkie 0 = martwe komórki)
    # Nanoszenie w oddalonych miejscach
    place_pattern(grid, "block", top=5, left=5)
    place_pattern(grid, "beehive", top=10, left=10)
    place_pattern(grid, "loaf", top=20, left=20)
    frames = [] # Lista przechowująca klatki GIF
    for epoch in range(epochs + 1):
        fname = f"stilllife_epoch_{epoch:02d}.png"
        save_grid_scaled(grid, fname, scale=10)
        frames.append((np.kron(grid, np.ones((10,10), dtype=np.uint8)) * 255).astype(np.uint8))
        if epoch < epochs:
            grid = game_of_life_step(grid)
    imageio.mimsave("three_still_life.gif", frames, duration=0.5)
    print("Wygenerowano three_still_life.gif")




def simulate_two_oscillators():
    """
    Symuluje dwa oscylatory (blinker i beacon) 
    - Oba oscylatory mają okres 2, więc 5 pełnych cykli = 10 epok.
    - Uruchamiamy iteracje, zbieramy klatki i zapisujemy 'two_oscillators.gif'.
    """
    N = 50
    epochs = 10
    grid = np.zeros((N, N), dtype=int)
    place_pattern(grid, "blinker", top=5,  left=5)
    place_pattern(grid, "beacon", top=20, left=30)
    frames = []
    for epoch in range(epochs + 1):
        fname = f"oscillator_epoch_{epoch:02d}.png"
        save_grid_scaled(grid, fname, scale=10)
        frames.append((np.kron(grid, np.ones((10,10), dtype=np.uint8)) * 255).astype(np.uint8))
        if epoch < epochs:
            grid = game_of_life_step(grid)
    imageio.mimsave("two_oscillators.gif", frames, duration=0.3)
    print("Wygenerowano two_oscillators.gif")


# Wyuchająca bomba, 120 epok
def simulate_rpentomino_explosion():
    N = 50
    epochs = 120
    grid = np.zeros((N, N), dtype=int)
    place_pattern(grid, "r_pentomino", top=20, left=20)
    frames = []
    for epoch in range(epochs + 1):
        fname = f"rpentomino_epoch_{epoch:03d}.png"
        save_grid_scaled(grid, fname, scale=10)
        frames.append((np.kron(grid, np.ones((10,10), dtype=np.uint8)) * 255).astype(np.uint8))
        if epoch < epochs:
            grid = game_of_life_step(grid)
    imageio.mimsave("rpentomino_explosion.gif", frames, duration=0.1)
    print("Wygenerowano rpentomino_explosion.gif")

# Metuzalech diehard, 100 epok
def simulate_diehard_methuselah():
    N = 50
    epochs = 100
    grid = np.zeros((N, N), dtype=int)
    place_pattern(grid, "diehard", top=10, left=10)
    frames = []
    for epoch in range(epochs + 1):
        fname = f"diehard_epoch_{epoch:03d}.png"
        fname = f"diehard_epoch_{epoch:03d}.png"
        save_grid_scaled(grid, fname, scale=10)
        frames.append((np.kron(grid, np.ones((10,10), dtype=np.uint8)) * 255).astype(np.uint8))
        if epoch < epochs:
            grid = game_of_life_step(grid)
    imageio.mimsave("diehard_methuselah.gif", frames, duration=0.1)
    print("Wygenerowano diehard_methuselah.gif")


if __name__ == "__main__":
    simulate_three_still_life()
    simulate_two_oscillators()
    simulate_rpentomino_explosion()
    simulate_diehard_methuselah()
   

