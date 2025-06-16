import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Wczytanie modelu JSON
with open('model.json') as f:
    model = json.load(f)

molecule = model['molecule']
params = model['parameters']
scenarios = model['scenarios']
reactions_def = model['reactions']

# Parsowanie
parsed_reactions = []
for entry in reactions_def:
    expr = entry['rate']
    stoich = np.array(entry['stoich'], float)
    code = compile(expr, '<string>', 'eval') # kompilowanie do obiektu kodu Pythona 
    def make_rate(c):
        return lambda y, flags: float(eval(c, {}, {**dict(zip(molecule, map(float, y))), **params, **flags}))
    parsed_reactions.append((make_rate(code), stoich))

# Gillespie podstawowy
def gillespie_basic(y0, reactions, flags, t_max):
    t, y = 0.0, np.array(y0, float) # aktualny czas i stan 
    ts, ys = [t], [y.copy()]        # przechowywanie historii czasu i stanu 
    while t < t_max:
        a = np.array([rate(y, flags) for rate, _ in reactions])   # a[i] to szybkość i-tej reakcji
        A = a.sum()
        if A <= 0:
            break
        # czas do kolejnego zdarzenia, rozkład jednostajny U(0,1) i transformacji odwrotnej 
        tau = -np.log(np.random.random()) / A
        cumulative = np.cumsum(a)
        r = np.searchsorted(cumulative, np.random.random() * A)
        # wybranie
        y = np.maximum(y + reactions[r][1], 0) # zmiana stanu 
        t += tau
        ts.append(t)
        ys.append(y.copy())
    return np.array(ts), np.vstack(ys)

# Tau-leap
def tau_leap_with_postleap(y0, reactions, flags, dt, steps):
    y = np.array(y0, float)
    traj = np.zeros((steps + 1, len(y0)))
    traj[0] = y
    V = np.vstack([stoich for _, stoich in reactions])
    original_dt = dt
    for i in range(1, steps + 1):
        accepted = False
        current_dt = dt
        while not accepted:
            a = np.array([rate(y, flags) for rate, _ in reactions]) # wektor szybkości 
            # losowanie liczby zdarzeń dla każdej reakcji ~ Poisson(a_j * current_dt) 
            k = np.random.poisson(a * current_dt)
            y_new = y + V.T @ k
            if np.all(y_new >= 0):
                # zaakceptowany skok
                y = y_new
                traj[i] = y
                accepted = True
                dt = original_dt # reset 
            else:
                # odrzucony skok, zmniejszyć current_dt, spróbować ponownie
                current_dt /= 2
                if current_dt < 1e-6:
                    # za mało – zaakceptuj bez zmiany
                    traj[i] = y
                    accepted = True
    return np.linspace(0, original_dt * steps, steps + 1), traj

# Parametry symulacji
t_max = 48 * 3600
dt = 300 #5 minut
steps = int(t_max / dt)

# Funkcja pomocnicza do obliczania pochodnych dla znajdowania stanu ustalonego
def equilibrium_residuals(y, flags):
    res = np.zeros_like(y, dtype=float)
    for rate_func, stoich in parsed_reactions:
        r = rate_func(y, flags)
        res += r * stoich
    return res

# Przygotowanie stanów równowagi metodą fsolve
scenario_names = list(scenarios.keys())
flagsA = scenarios[scenario_names[0]]
flagsB = scenarios[scenario_names[1]]
flagsC = scenarios[scenario_names[2]]
flagsD = scenarios[scenario_names[3]]

# fsolve do rozwiązania układu równań stanu ustalonego
guessA = [10000, 10000, 100000, 10000]
guessB = [5e6, 2e6, 1e4, 3e6]
guessC = [100000, 100000, 100000, 0]
guessD = [500000, 30000, 30000, 0]

solA = fsolve(lambda y: equilibrium_residuals(y, flagsA), guessA)
solB = fsolve(lambda y: equilibrium_residuals(y, flagsB), guessB)
solC = fsolve(lambda y: equilibrium_residuals(y, flagsC), guessC)
solD = fsolve(lambda y: equilibrium_residuals(y, flagsD), guessD)

initial_states = [solA, solB, solC, solD]
flags_list = [flagsA, flagsB, flagsC, flagsD]

display_titles = [
    "A - brak uszkodzeń DNA",
    "B - uszkodzone DNA",
    "C - nowotwór (PTEN off)",
    "D - terapia (siRNA on)"]

# Wybór algorytmu
alg_choice = input("Który algorytm chcesz uruchomić? Wpisz 'gillespie' lub 'tau': ").strip().lower()
simulation_function = gillespie_basic if alg_choice == 'gillespie' else tau_leap_with_postleap

# Rysunek trajektorii
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, title, flags, y0_init in zip(axes, display_titles, flags_list, initial_states):
    for rep in range(3):
        if alg_choice == 'tau':
            t, Y = simulation_function(y0_init, parsed_reactions, flags, dt, steps)
        else:
            t, Y = simulation_function(y0_init, parsed_reactions, flags, t_max)
        for idx, name in enumerate(molecule):
            ax.plot(t / 3600, Y[:, idx], alpha=0.6, label=name if rep == 0 else None)

    ax.set_title(f"{alg_choice.capitalize()}: {title}", pad=8)
    ax.set_xlabel('Czas [h]')
    ax.set_ylabel('Liczba cząsteczek')
    ax.legend(fontsize='small')
    ax.grid(True)

plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.25)
plt.show()
