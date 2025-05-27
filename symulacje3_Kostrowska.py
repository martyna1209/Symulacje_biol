import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp #metoda z SciPy do rozwiązywania układów ODE, zwraca obiekt sol (tablica sol.y)

#parametry i scenariusze (1=A, 3=C)
p_nom = dict(p1=8.8, p2=440.0, p3=100.0,
    d1=1.375e-14, d2=1.375e-4, d3=3e-5,
    k1=1.925e-4, k2=1e5, k3=1.5e5)
params = list(p_nom)
scenarios = {
    'A': {'sirna': False, 'pten_off': False, 'dna_ok': True},   #'A': zdrowe komórki bez uszkodzeń DNA 
    'C': {'sirna': False, 'pten_off': True, 'dna_ok': False}   #'C': komórki nowotworowe z uszkodzeniem DNA
}

#model ODE
def f(t, y, pvals, scen):
    p53, Mc, Mn, PTEN = y
    p2 = pvals['p2'] * (0.02 if scen['sirna'] else 1)
    p3 = 0.0 if scen['pten_off'] else pvals['p3']
    d2 = pvals['d2'] * (0.1 if scen['dna_ok'] else 1)
    reg = pvals['k3']**2 / (pvals['k3']**2 + PTEN**2) #własne uproszczenie
    return [
        pvals['p1'] - pvals['d1'] * p53 * Mn**2,
        p2 * p53**4/(p53**4 + pvals['k2']**4) - pvals['k1'] * reg * Mc - d2 * Mc,
        pvals['k1'] * reg * Mc - d2 * Mn,
        p3 * p53**4/(p53**4 + pvals['k2']**4) - pvals['d3'] * PTEN
    ]

#lokalna wrażliwość finite‐difference
def local_sensitivity(scen_key, delta=1e-3): #zwiększanie parametru o 0.1 %
    scen, pvals = scenarios[scen_key], p_nom.copy() #pobranie ustawień scenariusza i kopia parametrów
    t_span, t_eval = (0, 48*3600), np.linspace(0, 48*3600, 200) #zakres czasu 48h w sekundach i 200 punktów czasowych
    y0 = [1,1,1,1] #punkt początkowy

    sol0 = solve_ivp(f, t_span, y0, args=(pvals, scen), t_eval=t_eval) # rozwiązuje ODE z parametrami nominalnymi 
    base = sol0.y[0]  #wektor p53_base jako wartości p53 w kolejnych punktach czasowych            (n_stanów X n_punktów_czasowych): n_states=4 (p53, Mc, Mn, PTEN), n_times=200 
    S = np.zeros((len(params), len(t_eval))) #macierz wrażliwości każdy wiersz to inny parametr
    # każdemu parametrowi +delta
    for i, p in enumerate(params):
        pvals_ = pvals.copy()
        pvals_[p] *= 1 + delta
        sol = solve_ivp(f, t_span, y0, args=(pvals_, scen), t_eval=t_eval) #rozwiązuje ODE ponownie, już z parametrem powiększonym o delta
        pj = sol.y[0]  #wektor p53(t) po zmianie parametru

        # przybliżona pochodna ∂p53/∂p ≈ (pj – base)/(Δp)
        dYdp = (pj - base) / (pvals_[p] - p_nom[p])
        # normalizacja S = (∂y/∂p)*(p/y)       
        S[i] = dYdp * (p_nom[p] / base)
    return t_eval/3600, S # wektor czasu w godzinach i macierz wrażliwości



#Wykresy i bar‐chart ±20 %
if __name__ == '__main__':
    x0 = [1,1,1,1]
    for key in ['A','C']:
        
        t, S = local_sensitivity(key) # wektor czasu i macierz znormalizowana (parametr × czas)
        
        #ranking średniej
        mean_abs = np.mean(np.abs(S), axis=1)
        rank_avg = sorted(zip(params, mean_abs), key=lambda x: x[1], reverse=True)
        print("Ranking średniej |S| (0-48h):")
        for name, val in rank_avg:
            print(f"{name:>3}:{val:.3e}")

        #ranking |S| w chwili t=48h
        end_abs = np.abs(S[:, -1])
        rank_end = sorted(zip(params, end_abs), key=lambda x: x[1], reverse=True)
        print("\nRanking|S|(t=48h):")
        for name, val in rank_end:
            print(f"{name:>3}:{val:.3e}")

        # wybór top i bottom, średnia
        top_name, bot_name = rank_avg[0][0], rank_avg[-1][0]
        idx_top, idx_bot = params.index(top_name), params.index(bot_name)
        print(f"\nTop = {top_name}, Bot = {bot_name}")

        #krzywe funkcji wrażliwości
        plt.figure()
        plt.plot(t, S[idx_top], label=f"top = {top_name}")
        plt.plot(t, S[idx_bot], '--', label=f"bot = {bot_name}")
        plt.xlabel('czas [h]')
        plt.ylabel('znormalizowana wrażliwość p53')
        plt.title(f'Scenariusz {key}: funkcje wrażliwości')
        plt.legend(); plt.tight_layout(); plt.show()

        #obliczenia p53(48h) nominalnie
        base_val = float(solve_ivp(f, (0,48*3600), x0, args=(p_nom, scenarios[key]), t_eval=[48*3600]).y[0,-1])

        #dla każdego z top/bot liczymy ±20% 
        for name, idx in [(top_name, idx_top), (bot_name, idx_bot)]:
            #wektor względnych zmian
            deltas = []
            for fac in (0.8, 1.2):
                pv = p_nom.copy(); pv[name] *= fac
                val = float(solve_ivp(f, (0,48*3600), x0,
                                      args=(pv, scenarios[key]),
                                      t_eval=[48*3600]).y[0,-1])
                rel = (val - base_val)/base_val
                deltas.append(rel)
                print(f"{name}:{fac*100:.0f}% -> Δp53 = {rel:.4f}")
            #osobny bar-chart dla tego parametru
            fig, ax = plt.subplots()
            ax.bar(['-20%','+20%'], deltas, width=0.5)
            ax.axhline(0, color='black', linewidth=0.8)
            #skalowanie
            ymin, ymax = min(deltas), max(deltas)
            pad = (ymax-ymin)*0.2 if ymax!=ymin else 0.01
            ax.set_ylim(ymin-pad, ymax+pad)
            for i, v in enumerate(deltas):
                ax.text(i, v + np.sign(v)*pad*0.1, f"{v:.4f}",
                        ha='center', va='bottom' if v>=0 else 'top')
            ax.set_ylabel('Δp53')
            ax.set_title(f"{name} ±20% (scenariusz {key})")
            plt.tight_layout()
            plt.show()
        
    