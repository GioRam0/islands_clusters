#importo le librerie
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))

#importo il dataframe
ris_folder = os.path.join(cartella_corrente, "..", "results")
csv_path=os.path.join(ris_folder, 'analisys_df.csv')
df = pd.read_csv(csv_path)
output_folder = os.path.join(ris_folder, f"qq_plot_distri_coda")
os.makedirs(output_folder, exist_ok=True)

#colonne per cui realizzare i Q-Q plot
colonne=['superficie_res', 'Densità_pop', 'eolico', 'offshore', 'geothermal_potential', 'gdp_cons_pop_urban_merged', 'hydro', 'solar_seas_ind']
#distribuzioni da confrontare
distributions = {
    'Esponenziale': stats.expon,
    'Gamma': stats.gamma,
    'Weibull': stats.weibull_min,
    'Lognormale': stats.lognorm
}
#itero per le features
for col in colonne:
    x=df[(df[col]>0)][col]
    if col=='gdp-pop-urban_merged':
        min=df[col].min()
        x=df[col]-min+1
    plt.figure(figsize=(20, 16))
    for i, (name, dist) in enumerate(distributions.items(), 1):
        plt.subplot(2, 2, i)
        params = dist.fit(x)
        #qq plot con la distribuzione stimata
        stats.probplot(x, dist=dist, sparams=params[:-2], plot=plt)
        plt.title(f'Q-Q Plot: {name}')
        plt.xlabel('Quantili teorici')
        plt.ylabel('Quantili campionari')
    plt.tight_layout()
    output_path=os.path.join(output_folder, f"qq1_{col}.png")
    plt.savefig(output_path)
    plt.close() 