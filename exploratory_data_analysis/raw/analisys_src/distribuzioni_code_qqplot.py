import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))

#importo il dataframe
ris_folder = os.path.join(cartella_corrente, "..", "risultati")
pkl_path=os.path.join(ris_folder, 'analisys_df.pkl')
output_folder = os.path.join(ris_folder, f"qq_plot_distri_coda")
os.makedirs(output_folder, exist_ok=True)
df = pd.read_pickle(pkl_path)
colonne=['IslandArea', 'Popolazione', 'Densità_pop', 'eolico', 'offshore', 'gdp', 'gdp_pro_capite', 'geothermal_potential', 'hydro', 'urban_area', 'urban_area_rel', 'ele_max']
distributions = {
    'Esponenziale': stats.expon,
    'Gamma': stats.gamma,
    'Weibull': stats.weibull_min,
    'Lognormale': stats.lognorm
}
for col in colonne:
    print(f'isole con dati >0 per {col}: {len(df[(df[col]>0)])}')
    x=df[(df[col]>0)][col]
    plt.figure(figsize=(20, 16))
    for i, (name, dist) in enumerate(distributions.items(), 1):
        plt.subplot(2, 2, i)
        params = dist.fit(x)
        # Q-Q plot con la distribuzione stimata
        stats.probplot(x, dist=dist, sparams=params[:-2], plot=plt)
        plt.title(f'Q-Q Plot: {name}')
        plt.xlabel('Quantili teorici')
        plt.ylabel('Quantili campionari')
    plt.tight_layout()
    output_path=os.path.join(output_folder, f"qq1_{col}.png")
    plt.savefig(output_path)
    plt.close()

#isole con dati >0 per IslandArea: 2736
#isole con dati >0 per Popolazione: 2736
#isole con dati >0 per Densità_pop: 2736
#isole con dati >0 per eolico: 2736
#isole con dati >0 per offshore: 1276
#isole con dati >0 per gdp: 2736
#isole con dati >0 per gdp_pro_capite: 2736
#isole con dati >0 per geothermal_potential: 375
#isole con dati >0 per hydro: 830
#isole con dati >0 per urban_area: 510
#isole con dati >0 per urban_area_rel: 510
#isole con dati >0 per ele_max: 2723