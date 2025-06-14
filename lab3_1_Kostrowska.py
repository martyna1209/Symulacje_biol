import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Parametry i scenariusze
p_nom = dict(p1=8.8,      
    p2=440.0,    
    p3=100.0,   
    d1=1.375e-14,
    d2=1.375e-4,
    d3=3e-5,
    k1=1.925e-4,
    k2=1e5,
    k3=1.5e5)

scenarios = {'A': {'sirna': False, 'pten_off': False, 'dna_ok': True},   # zdrowy
    'C': {'sirna': False, 'pten_off': True,  'dna_ok': False}   # nowotworowy
}

# Układ różniczkowy
def model(y, p, scen):
    p53, Mc, Mn, PTEN = y
    p2 = p['p2'] * (0.02 if scen['sirna'] else 1.0)
    p3 = 0.0 if scen['pten_off'] else p['p3']
    d2 = p['d2'] * (0.1 if scen['dna_ok'] else 1.0)
    reg = p['k3']**2 / (p['k3']**2 + PTEN**2)
    return np.array([
        p['p1'] - p['d1'] * p53 * Mn**2,
        p2 * p53**4 / (p53**4 + p['k2']**4) - p['k1']*reg*Mc - d2*Mc,
        p['k1']*reg*Mc - d2*Mn,
        p3 * p53**4 / (p53**4 + p['k2']**4) - p['d3']*PTEN])

def equilibrium_residuals(y, p, scen):
    return model(y, p,scen)

# Punkty równowagi
equilibria = {}
for label, scen in scenarios.items():
    guess = np.array([2e4, 1e4, 1.5e5, 1e4])
    sol = fsolve(lambda y: equilibrium_residuals(y, p_nom, scen), guess)
    equilibria[label] = sol
    print(f"Equilibrium for scenario {label}: {sol}")

# RK4, przebieg czasu
def rk4_step(y, pvals, scen, h):
    k1 = model(y, pvals, scen)
    k2 = model(y + 0.5*h*k1, pvals, scen)
    k3 = model(y + 0.5*h*k2, pvals, scen)
    k4 = model(y + h*k3,pvals, scen)
    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

def run_sim_time_series(pvals, scen, y0, t_end=3600*48, steps=200):
    """
    Symulacja RK4 od 0 do t_end (48 h = 172800 s),
    zwracana tablica kształtu (steps+1,) w tym poziom p53 w kolejnych krokach.
    """
    y = y0.copy().astype(float)
    h = t_end/steps
    ts = np.zeros(steps+1)
    ts[0] = y[0]
    for i in range(1, steps+1):
        y = rk4_step(y, pvals, scen, h)
        ts[i] = y[0]
    return ts
def sample_parameters(p_base):
    return {k: np.random.uniform(0.8*v, 1.2*v) for k, v in p_base.items()}

# Globalna analiza Sobola
def global_sensitivity(p_base, scen, y0, N=500, steps=200):
    """
    Zwracane:
      -S1_final, ST_final: słowniki indeksów pierwszego i całkowitego w t końcowym
      -S1_avg, ST_avg: indeksy "średnie" uśrednione po wszystkich krokach czasowych
    """
    param_list = list(p_base.keys())
    k = len(param_list)
    # tablice dla wyników A,B i hybryd
    Y_A = np.zeros((N, steps+1))
    Y_B = np.zeros((N, steps+1))
    Y_AB = np.zeros((k, N, steps+1))

    for i in range(N):
        A = sample_parameters(p_base)
        B = sample_parameters(p_base)
        Y_A[i] = run_sim_time_series(A, scen, y0, steps=steps)
        Y_B[i] = run_sim_time_series(B, scen, y0, steps=steps)
        for j,key in enumerate(param_list):
            ABj = A.copy()
            ABj[key] = B[key]
            Y_AB[j,i] = run_sim_time_series(ABj, scen, y0, steps=steps)

    # Wariancje globalne w każdym kroku
    D_t = np.var(np.concatenate((Y_A, Y_B), axis=0), axis=0, ddof=1)  # shape (steps+1,)
    # Macierze na indeksy
    S1_t  = np.zeros((k, steps+1))
    ST_t  = np.zeros((k, steps+1))
    for j in range(k):
        D_j_t = np.mean(Y_B * (Y_AB[j] - Y_A), axis=0)
        D_jt_t = 0.5*np.mean((Y_A - Y_AB[j])**2, axis=0)
        S1_t[j] = D_j_t / D_t
        ST_t[j] = D_jt_t / D_t

    # W chwili ostatniej (ostatni indeks)
    S1_final = {param_list[j]: S1_t[j,-1] for j in range(k)}
    ST_final = {param_list[j]: ST_t[j,-1] for j in range(k)}
    # Średnio po wszystkich krokach
    S1_avg = {param_list[j]: np.mean(S1_t[j]) for j in range(k)}
    ST_avg = {param_list[j]: np.mean(ST_t[j]) for j in range(k)}
    # kontrola sumy S1_final
    if sum(S1_final.values()) > 1+1e-6:
        print(f"WARNING: sum(S1_final)={sum(S1_final.values()):.3f} > 1 – Warto rozważyć zwiększenie N.")
    return S1_final, ST_final, S1_avg, ST_avg

# Rankingi 
if __name__ == '__main__':
    np.random.seed(0)
    N = 500  
    steps = 200
    results = {}
    for label, scen in scenarios.items():
        y0 = equilibria[label]
        S1_fin, ST_fin, S1_avg, ST_avg = global_sensitivity(p_nom, scen, y0, N=N, steps=steps)
        results[label] = {
            'final': (S1_fin, ST_fin),
            'avg': (S1_avg, ST_avg)}
        # Ranking pierwszego rzędu i total
        def print_rank(dct, title):
            print(f"\nScenariusz {label} – {title}")
            for k,v in sorted(dct.items(), key=lambda x: -x[1]):
                print(f"{k:>4}: {v:.3f}")
        print_rank(S1_fin, "S1 (t=48h)")
        print_rank(ST_fin, "STot (t=48h)")
        print_rank(S1_avg, "S1 średnio 0-48h")
        print_rank(ST_avg, "STot średnio 0-48h")

    # Wykresy 
    params = list(p_nom.keys())
    x = np.arange(len(params))
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for row, (tp, lbl) in enumerate([('final','t=48h'), ('avg','średnio')]):
        for col, label in enumerate(scenarios):
            S1, ST = results[label][tp]
            ax = axs[row, col]
            ax.bar(x-0.2, [S1[p] for p in params], width=0.4, label='$S_i$')
            ax.bar(x+0.2, [ST[p] for p in params], width=0.4, label='$S_i^{tot}$')
            ax.set_xticks(x); ax.set_xticklabels(params, rotation=45)
            ax.set_title(f"Scenariusz {label}, {lbl}")
            ax.set_ylim(0,1); ax.legend()
    plt.tight_layout()
    plt.show()
