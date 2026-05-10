# ============================================
# Adsorption kinetic modeling
# Models:
# - Pseudo-first-order (PFO)
# - Pseudo-second-order (PSO)
# - Elovich
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --------------------------------------------
# Time (minutes)
# --------------------------------------------
t = np.array([ ], dtype=float)

# --------------------------------------------
# Experimental data (q(t))
# --------------------------------------------
qt_20 = np.array([ ])

qt_30 = np.array([ ])

qt_40 = np.array([ ])

qt_50 = np.array([ ])

data = {20: qt_20, 30: qt_30, 40: qt_40, 50: qt_50}

# --------------------------------------------
# Kinetic models
# --------------------------------------------

# Pseudo-first-order
def pfo(t, qe, k1):
    return qe * (1 - np.exp(-k1 * t))

# Pseudo-second-order
def pso(t, qe, k2):
    return (k2 * qe**2 * t) / (1 + k2 * qe * t)

# Elovich model
def elovich(t, alpha, beta):
    return (1/beta) * np.log(1 + alpha * beta * t)

# --------------------------------------------
# R² function
# --------------------------------------------
def calc_r2(y_exp, y_fit):
    ss_res = np.sum((y_exp - y_fit)**2)
    ss_tot = np.sum((y_exp - np.mean(y_exp))**2)
    return 1 - ss_res / ss_tot

# --------------------------------------------
# Fitting loop
# --------------------------------------------
results = []

for T, qt_exp in data.items():

    plt.figure(figsize=(7, 5))

    # ---------------- PFO ----------------
    try:
        popt_pfo, _ = curve_fit(pfo, t, qt_exp,
                               p0=[np.max(qt_exp), 0.1],
                               maxfev=10000)
        qt_pfo = pfo(t, *popt_pfo)
        r2_pfo = calc_r2(qt_exp, qt_pfo)
    except:
        popt_pfo = [np.nan, np.nan]
        r2_pfo = np.nan

    # ---------------- PSO ----------------
    try:
        popt_pso, _ = curve_fit(pso, t, qt_exp,
                               p0=[np.max(qt_exp), 0.01],
                               maxfev=10000)
        qt_pso = pso(t, *popt_pso)
        r2_pso = calc_r2(qt_exp, qt_pso)
    except:
        popt_pso = [np.nan, np.nan]
        r2_pso = np.nan

    # ---------------- Elovich ----------------
    try:
        popt_elo, _ = curve_fit(elovich, t, qt_exp,
                               p0=[1, 1],
                               maxfev=10000)
        qt_elo = elovich(t, *popt_elo)
        r2_elo = calc_r2(qt_exp, qt_elo)
    except:
        popt_elo = [np.nan, np.nan]
        r2_elo = np.nan

    # Save results
    results.append({
        'Temperature (°C)': T,

        'qe_PFO': popt_pfo[0],
        'k1 (1/min)': popt_pfo[1],
        'R2_PFO': r2_pfo,

        'qe_PSO': popt_pso[0],
        'k2 (g/mg·min)': popt_pso[1],
        'R2_PSO': r2_pso,

        'alpha_Elovich': popt_elo[0],
        'beta_Elovich': popt_elo[1],
        'R2_Elovich': r2_elo
    })

    # ------------------------------------
    # Plot comparison
    # ------------------------------------
    plt.scatter(t, qt_exp, color='black', label='Experimental')

    if not np.isnan(r2_pfo):
        plt.plot(t, qt_pfo, '--', label='PFO')

    if not np.isnan(r2_pso):
        plt.plot(t, qt_pso, '-.', label='PSO')

    if not np.isnan(r2_elo):
        plt.plot(t, qt_elo, ':', label='Elovich')

    plt.xlabel('Time (min)')
    plt.ylabel('q(t)')
    plt.title(f'Kinetic models at {T} °C')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --------------------------------------------
# Final results table
# --------------------------------------------
results_df = pd.DataFrame(results).sort_values('Temperature (°C)')

print("\nKinetic model comparison:")
print(results_df)