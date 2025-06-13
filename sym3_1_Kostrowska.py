import json  
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp

# Wczytanie modelu JSON 
with open('model.json') as f:
    model = json.load(f)

molecule = model['molecule'] 
params = model['parameters'] 
scenarios = model['scenarios'] 
reactions_def = model['reactions'] 
y0 = np.array(model['initial_state'], float)

# Parsowanie 
parsed_reactions = [] 
for entry in reactions_def:
    expr = entry['rate'] 
    stoich = np.array(entry['stoich'], float) 
    code = compile(expr, '<string>', 'eval') # kompilowanie do obiektu kodu Pythona 
    def make_rate(c): 
        return lambda y, flags: eval(c, {}, {**dict(zip(molecule, y)), **params, **flags}) 
    parsed_reactions.append((make_rate(code), stoich)) 

# Funkcja do układu deterministycznego ODE
def deterministic_rhs(t, y, flags):
    dydt = np.zeros_like(y)
    for rate_func, nu in parsed_reactions:
        dydt += rate_func(y, flags) * nu
    return dydt

# Wyznaczenie stanu ustalonego z ODE
def find_steady_state(y_init, flags, t_end=1e6):
    sol = solve_ivp(lambda t, y: deterministic_rhs(t, y, flags), [0, t_end], y_init, method='RK45', atol=1e-8, rtol=1e-6)
    return sol.y[:, -1]

# Gillespie podstawowy 
def gillespie_basic(y0, reactions, flags, t_max): 
    t, y = 0.0, np.array(y0, float) # aktualny czas i stan 
    ts, ys = [t], [y.copy()]        # przechowywanie historii czasu i stanu 
    while t < t_max:
        a = np.array([rate(y, flags) for rate, _ in reactions]) # a[i] to szybkość i-tej reakcji 
        A = a.sum()
        if A <= 0:
            break
        # czas do kolejnego zdarzenia, rozkład jednostajny U(0,1) i transformacji odwrotnej
        u1 = np.random.random()
        tau = -np.log(u1) / A
        # losowanie, która reakcja zajdzie druga liczba losowa z U(0,1) 
        u2 = np.random.random() * A
        cumulative = 0.0
        for i, ai in enumerate(a):
            cumulative += ai
            if u2 <= cumulative:
                r = i
                break
        # wybranie
        y = np.maximum(y + reactions[r][1], 0)
        t += tau
        ts.append(t); ys.append(y.copy())
    return np.array(ts), np.vstack(ys)

# Tau-leap 
def tau_leap_with_postleap(y0, reactions, flags, dt, steps): 
    y = np.array(y0, float)
    traj = np.zeros((steps+1, len(y0)))
    traj[0] = y
    V = np.vstack([stoich for _, stoich in reactions])
    original_dt = dt
    for i in range(1, steps+1): 
        accepted = False 
        current_dt = dt 
        while not accepted: 
            a = np.array([rate(y, flags) for rate, _ in reactions]) # wektor szybkości 
            k = np.random.poisson(a * current_dt) 
            y_new = y + V.T.dot(k)
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
    return np.linspace(0, original_dt * steps, steps+1), traj

# Parametry symulacji 
t_max = 48 * 3600
dt = 300
steps = int(t_max / dt)

flags_list = list(scenarios.values()) 
initial_states = [find_steady_state(y0, flags) for flags in flags_list]

# Wizualizacja wyników
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
display_titles = [
    "A - brak uszkodzeń DNA", "B - uszkodzone DNA", 
    "C - nowotwór (PTEN off)", "D - terapia (siRNA on)"]

for ax, title, flags, y0_init in zip(axes, display_titles, flags_list, initial_states):
    for rep in range(3): 
        t, Y = tau_leap_with_postleap(y0_init, parsed_reactions, flags, dt, steps)
        for idx, name in enumerate(molecule): 
            ax.plot(t/3600, Y[:, idx], alpha=0.6, label=name if rep == 0 else None) 
    ax.set_title(f"Tau-leap: {title}", pad=8)
    ax.set_xlabel('Czas [h]') 
    ax.set_ylabel('Liczba cząsteczek') 
    ax.legend(fontsize='small') 
    ax.grid(True)

plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.25)
plt.show()
