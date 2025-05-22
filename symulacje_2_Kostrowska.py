import json
import numpy as np
import matplotlib.pyplot as plt

# Wczytanie modelu JSON
with open('model.json') as f:
    model = json.load(f)

molecule = model['molecule']
params = model['parameters']
scenarios = model['scenarios']
reactions_def = model['reactions']
y0 = model['initial_state']

# Parsowanie
parsed_reactions = []
for entry in reactions_def:
    expr = entry['rate']
    stoich = np.array(entry['stoich'], float)
    code = compile(expr, '<string>', 'eval') #kompilowanie do obiektu kodu Pythona
    def make_rate(c):
        return lambda y, flags: eval(c, {}, {**dict(zip(molecule, y)), **params, **flags}) #lambda(y, flags) mapuje y na nazwy zmiennych, dokleja parametry z params i flagi, wywołuje eval()
    parsed_reactions.append((make_rate(code), stoich))

# Gillespie podstawowy
def gillespie_basic(y0, reactions, flags, t_max): #y0 = stan początkowy    #reactions = (szybkość z warunkami flags, zmiana_stanu) #flags  to np. dna_ok, czyli dodatkowe warunki
    t, y  = 0.0, np.array(y0, float) #aktualny czas i stan
    ts, ys = [t], [y.copy()] #przechowywanie t i y
    # każda reakcja: rate(y,flags), wektor stoich
    while t < t_max:
        a = np.array([rate(y, flags) for rate, _ in reactions]) # a[i] to szybkość i-tej reakcji przy obecnym stanie y
        A = a.sum() #A = Σ ai
        if A <= 0: break
        tau = np.random.exponential(1/A) # losowanie czasu do kolejnego zdarzenia: τ ~ Exp(A)
        r = np.random.choice(len(a), p=a/A) # która reakcja zajdzie
        y = np.maximum(y + reactions[r][1], 0) # zmiana stanu y
        t += tau # zmiana czas
        ts.append(t); ys.append(y.copy()) # zapis nowych wartości
    return np.array(ts), np.vstack(ys)

# Tau-leap
def tau_leap_with_postleap(y0, reactions, flags, dt, steps): #dt to początkowy krok, steps- ilość kroków
    y = np.array(y0, float)
    traj = np.zeros((steps+1, len(y0))) #macierz trajektori i wiersze to kolejne kroki, kolumny to zmienne
    traj[0] = y #zapis stanu w t=0
    V = np.vstack([stoich for _, stoich in reactions])
    original_dt = dt  #zachować oryginalną wartość dt do resetu po odrzuceniu

    # Akceptacja skoku
    for i in range(1, steps+1):
        accepted = False
        current_dt = dt
        
        while not accepted:
            a = np.array([rate(y, flags) for rate, _ in reactions]) # wektor prędkości
            # losowanie
            k = np.random.poisson(a * current_dt) # losowanie ~ Poisson(a_j * current_dt)
            y_new = y + V.T.dot(k) #nowy stan y_new = y + Σ_j stoich_j * k_j
            if np.all(y_new >= 0):
                # zaakceptowany skok, aktualizacja i zapis stanu
                y = y_new 
                traj[i] = y
                accepted = True
                dt = original_dt  # reset do pierwotnego dt
            else:
                # odrzucony skok, zmniejszyć current_dt, spróbować ponownie
                current_dt /= 2
                if current_dt < 1e-6:
                    # za mało – zaakceptuj bez zmiany
                    traj[i] = y
                    accepted = True

    return np.linspace(0, original_dt*steps, steps+1), traj
# Parametry symulacji
t_max = 48 * 3600   # 48 h w sekundach
dt = 300         # 5 minut
steps = int(t_max / dt)

# τ-leap dla wszystkich scenariuszy (2×2)
flags_list = list(scenarios.values())
display_titles = [
    "A - brak uszkodzeń DNA",
    "B - uszkodzone DNA",
    "C - nowotwór (PTEN off)",
    "D - terapia (siRNA on)",]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, title, flags in zip(axes, display_titles, flags_list):
    for rep in range(3):
        t, Y = tau_leap_with_postleap(y0, parsed_reactions, flags, dt, steps)
        for idx, name in enumerate(molecule):
            ax.plot(t/3600, Y[:, idx],
                    alpha=0.6,
                    label=name if rep == 0 else None)
    ax.set_title(f"Tau-leap: {title}", pad=8)  
    ax.set_xlabel('Czas [h]')
    ax.set_ylabel('Liczba cząsteczek')
    ax.legend(fontsize='small')
    ax.grid(True)

plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.25)
plt.show()


