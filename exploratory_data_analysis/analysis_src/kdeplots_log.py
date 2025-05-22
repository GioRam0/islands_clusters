import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..")

#importo il dataframe
ris_folder = os.path.join(cartella_corrente, "..", "risultati")
pkl_path=os.path.join(ris_folder, 'analisys_df.pkl')
df = pd.read_pickle(pkl_path)
#colonne con distribuzioni a coda
colonne_code=['IslandArea', 'Popolazione', 'Densità_pop', 'eolico','offshore', 'gdp','gdp_pro_capite', 'geothermal_potential', 'hydro', 'urban_area', 'urban_area_rel','ele_max']

#creo ed esporto i kdeplot
output_folder = os.path.join(ris_folder, "kde_plots")
os.makedirs(output_folder, exist_ok=True)
for col in df[colonne_code].select_dtypes(include='number').columns:
    output_folder1 = os.path.join(output_folder, "logaritmi")
    os.makedirs(output_folder1, exist_ok=True)
    output_path = os.path.join(output_folder1, f"{col}_kdeplot.png")
    plt.figure(figsize=(10, 15))
    data=np.log(df[df[col] != 0][col])
    sns.kdeplot(data, shade=True, color="skyblue",fill=True)
    plt.title(f"KDE Plot di {col}")
    plt.xlabel(col)
    plt.ylabel("Densità")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()