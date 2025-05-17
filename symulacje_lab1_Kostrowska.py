import numpy as np
import matplotlib.pyplot as plt

# Stałe biologiczne wynikające z modelu (zgodne z notatką) 
p1, p2_base, p3_base = 8.8, 440, 100    # produkcja: p53, MDM, PTEN
d1, d2_base, d3 = 1.375e-14, 1.375e-4, 3e-5     #degradacje: p53, MDM, PTEN
k1, k2, k3 = 1.925e-4, 1e5, 1.5e5   #parametry regulacji

# Scenariusze symulacji 
scenarios = {
    "A – brak uszkodzeń DNA, PTEN on, brak siRNA": {"sirna": False, "pten_off": False, "dna_ok": True},
    "B – uszkodzone DNA, PTEN on, brak siRNA": {"sirna": False, "pten_off": False, "dna_ok": False},
    "C – nowotwór (PTEN off, DNA uszkodzone, brak siRNA)": {"sirna": False, "pten_off": True, "dna_ok": False},
    "D – terapia (PTEN off, DNA uszkodzone, obecny siRNA)": {"sirna": True, "pten_off": True, "dna_ok": False},
}

# sirna=True to silne zahamowanie MDM (p2 *= 0.02)
# pten_off=True to wyłączenie produkcji PTEN (p3 = 0)
# dna_ok=True to obniżenie degradacji MDM (d2 *= 0.1)
# Równania różniczkowe
def dY(y, sirna, pten_off, dna_ok):
    p53, M_c, M_n, PTEN = y

    # Modyfikacja parametrów wg scenariusza
    p2 = p2_base * (0.02 if sirna else 1.0)
    p3 = 0.0 if pten_off else p3_base
    d2 = d2_base * (0.1 if dna_ok else 1.0)
    reg = k3**2 / (k3**2 + PTEN**2) # Regulacja przez PTEN (hamuje transport MDM do jądra)

    # Równania ODE
    dp53 = p1 - d1 * p53 * M_n**2
    dMc = p2 * p53**4 / (p53**4 + k2**4) - k1 * reg * M_c - d2 * M_c
    dMn = k1 * reg * M_c - d2 * M_n
    dP = p3 * p53**4 / (p53**4 + k2**4) - d3 * PTEN

    return np.array([dp53, dMc, dMn, dP]) # zwraca macierz pochodnych wszystkich zmiennych

# Jeden krok RK4, średnia na podstawie 4 wyliczeń pochodnych
def rk4_step(fun, y, h, **flags): #** flags to słownik argumentów
    k1 = fun(y, **flags)
    k2 = fun(y + k1 * (h/2), **flags)
    k3 = fun(y + k2 * (h/2), **flags)
    k4 = fun(y + k3 * h, **flags)
    return y + (k1 + 2*k2 + 2*k3 + k4) * (h/6) # klasyczny wzór RK4

# Parametry czasowe
dt = 6    # krok czasowy: 6 minut - płynne
T_tot = 48 * 60   # 48 godzin w minutach
times = np.arange(0, T_tot + dt, dt) # wektor czasu [0, 6, 12, .., 2880]

# Warunki początkowe wybrane przez użytkownika
y0 = np.array([0.0, 2000.0, 15000.0, 2000.0])  # [p53, MDM_cyt, MDM_nukl, PTEN]

# Symulacja i wykresy 
for title, flags in scenarios.items():
    Y = np.zeros((len(times), 4)) # macierz wyników: wiersze =czas, kolumny = zmienne
    Y[0] = y0.copy() #wartości początkowe jako pierwszy wiersz

    for i in range(len(times) - 1):
        Y[i+1] = rk4_step(dY, Y[i], dt, **flags) # RK4 dla każdego kroku czasowego

    # Wykres
    plt.figure(figsize=(8, 5))
    plt.plot(times / 60, Y[:, 0], label='p53')
    plt.plot(times / 60, Y[:, 1], label='MDM cyt')
    plt.plot(times / 60, Y[:, 2], label='MDM nukl')
    plt.plot(times / 60, Y[:, 3], label='PTEN')

    plt.title(f"Symulacja w czasie 48h: {title}")
    plt.xlabel("Czas [h]")
    plt.ylabel("Stężenie [nM] / ilość")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
