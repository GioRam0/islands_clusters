import numpy as np
import scipy.stats as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, probplot, normaltest, lognorm

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..")

#importo il dataframe
ris_folder = os.path.join(cartella_corrente, "risultati")
pkl_path=os.path.join(ris_folder, 'analisys_df.pkl')
df = pd.read_pickle(pkl_path)

#colonne di cui voglio trovare una distribuzuione
colonne=['IslandArea', 'Popolazione', 'Densità_pop', 'eolico', 'offshore', 'gdp', 'gdp_pro_capite', 'geothermal_potential', 'hydro', 'urban_area', 'urban_area_rel', 'ele_max']
print(len(df[(df['eolico']==0)]))
log_eolico = np.log(np.log(df[(df['eolico']!=0)]['eolico']))

# Test di Shapiro-Wilk
shapiro_stat, shapiro_p = shapiro(log_eolico)
print(f"Shapiro-Wilk p-value: {shapiro_p:.4f}")

# Test di D’Agostino e Pearson
normaltest_stat, normaltest_p = normaltest(log_eolico)
print(f"D’Agostino and Pearson p-value: {normaltest_p:.4f}")
print(sksn)
# Esempio: normal, lognormal, exponential, prova amche con altre distribuzioni
dists = {
    'norm': st.norm,
    'lognorm': st.lognorm,
    'expon': st.expon
}

results = []
for name, dist in dists.items():
    #calcola i parametri di forma piu verosimili quali media deviazione ecc.
    params = dist.fit(data)
    # KS‐test contro la stessa distribuzione con quei parametri
    #prende in input i dati, il nome della distribuzione e i parametri e restituisce statistiche di somiglianza
    #D basso buona aderenza
    #p basso, fifiuti la distri
    D, p = st.kstest(data, name, args=params)
    # AIC approssimativo: 2*k - 2*logL, usando logpdf
    #calcola la logsomiglianza, vogliamo aic basso
    k = len(params)
    logL = np.sum(dist.logpdf(data, *params))
    AIC = 2*k - 2*logL
    #crea una tupla con i vari valori
    results.append((name, params, D, p, AIC))

# Ordina per AIC
results.sort(key=lambda x: x[4])
for r in results:
    print(f"{r[0]:8s}  KS‐D={r[2]:.3f}, p={r[3]:.3f}, AIC={r[4]:.1f}")


#distri con code lunghe
import powerlaw

fit = powerlaw.Fit(data)
print("alpha (esponente Pareto):", fit.alpha)
print("xmin (soglia code):", fit.xmin)
R, p = fit.distribution_compare('power_law', 'lognormal')
print("Confronto power_law vs lognormal:", R, p)

#per gli outliers fai valutazioni sulle singole distribuzioni