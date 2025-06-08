import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))

#importo il dataframe
ris_folder = os.path.join(cartella_corrente, "..", "risultati")
pkl_path=os.path.join(ris_folder, 'analisys_df.pkl')
df = pd.read_pickle(pkl_path)
#colonne con distribuzioni a coda
colonne_code=['IslandArea', 'Densità_pop', 'offshore', 'geothermal_potential', 'gdp_pop_urban_merged', 'hydro', 'solar_seas_ind']

#creo ed esporto i kdeplot
output_folder = os.path.join(ris_folder, "kde_plots")
os.makedirs(output_folder, exist_ok=True)
for col in df[colonne_code].select_dtypes(include='number').columns:
    output_folder1 = os.path.join(output_folder, "logaritmi")
    os.makedirs(output_folder1, exist_ok=True)
    output_path = os.path.join(output_folder1, f"{col}_kdeplot.png")
    plt.figure(figsize=(10, 15))
    min=df[col].min()
    data=np.log1p(df[col]-min)
    if col=='geothermal_potential' or col=='offshore' or col=='hydro':
        data=np.log1p(df[df[col]>0][col])
    sns.kdeplot(data, shade=True, color="skyblue",fill=True)
    plt.title(f"KDE Plot di {col}")
    plt.xlabel(col)
    plt.ylabel("Densità")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()