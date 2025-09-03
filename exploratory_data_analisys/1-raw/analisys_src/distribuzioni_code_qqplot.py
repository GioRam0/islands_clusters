#importo le librerie
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))

#importo il dataframe
df_folder = os.path.join(cartella_corrente, "..", "..")
csv_path=os.path.join(df_folder, 'df_raw.csv')
df = pd.read_csv(csv_path)

#cartella esportazione risultati
ris_folder = os.path.join(cartella_corrente, "..", "results")
os.makedirs(ris_folder, exist_ok=True)
output_folder = os.path.join(ris_folder, f"qq_plot_distri_coda")
os.makedirs(output_folder, exist_ok=True)

#colonne per cui realizzare i Q-Q plot
colonne=['IslandArea', 'Popolazione', 'Densità_pop', 'eolico', 'offshore', 'gdp_2019', 'consumption', 'geothermal_potential', 'hydro', 'urban_area', 'urban_area_rel', 'ele_max']
#distribuzioni da confrontare
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
        #Q-Q plot con la distribuzione stimata
        stats.probplot(x, dist=dist, sparams=params[:-2], plot=plt)
        plt.title(f'Q-Q Plot: {name}')
        plt.xlabel('Quantili teorici')
        plt.ylabel('Quantili campionari')
    plt.tight_layout()
    output_path=os.path.join(output_folder, f"qq1_{col}.png")
    plt.savefig(output_path)
    plt.close()

#isole con dati >0 per IslandArea: 2012
#isole con dati >0 per Popolazione: 2012
#isole con dati >0 per Densità_pop: 2012
#isole con dati >0 per eolico: 2012
#isole con dati >0 per offshore: 899
#isole con dati >0 per gdp_2019: 2012
#isole con dati >0 per consumption: 2012
#isole con dati >0 per geothermal_potential: 289
#isole con dati >0 per hydro: 762
#isole con dati >0 per urban_area: 434
#isole con dati >0 per urban_area_rel: 434
#isole con dati >0 per ele_max: 2003