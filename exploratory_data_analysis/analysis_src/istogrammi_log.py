import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..")

#importo il dataframe
ris_folder = os.path.join(cartella_corrente, "..", "risultati")
pkl_path=os.path.join(ris_folder, 'analisys_df.pkl')
df = pd.read_pickle(pkl_path)
#colonne con distribuzioni a coda
colonne_code=['IslandArea', 'Popolazione', 'Densità_pop', 'offshore', 'gdp', 'geothermal_potential', 'hydro', 'urban_area', 'urban_area_rel','ele_max']

#creo ed esporto gli istogrammi
output_folder = os.path.join(ris_folder, "istogrammi")
os.makedirs(output_folder, exist_ok=True)
for col in df[colonne_code].select_dtypes(include='number').columns:
    output_folder1 = os.path.join(output_folder, "logaritmi")
    os.makedirs(output_folder1, exist_ok=True)
    output_path = os.path.join(output_folder1, f"{col}_istogramma.png")
    plt.figure(figsize=(10, 15))
    data=np.log(df[df[col] != 0][col])
    data.hist(bins=60, color='skyblue', edgecolor='black')
    plt.title(f"Istogramma di logaritmo di {col}")
    plt.xlabel(col)
    plt.ylabel("Frequenza")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

#per gdp_pro_capite faccio l'isogramma con colori diversi in base all'etichetta
#dizionario con le etichette possibili e i colori che voglio loro associare nell'istogramma
labels={'XS': 'red', 'S': 'green', 'M':'yellow', 'L': 'blue'}
output_folder1 = os.path.join(output_folder, "logaritmi")
os.makedirs(output_folder1, exist_ok=True)
output_path = os.path.join(output_folder1, f"{col}_istogramma.png")
plt.figure(figsize=(10, 15))
df_log=df[['gdp_pro_capite','GDP_procap_etichetta']]
df_log=df_log[df_log['gdp_pro_capite'] != 0]
df_log['gdp_pro_capite']=np.log(df_log['gdp_pro_capite'])
#imposto la larghezza delle varie colonne
bin_width=(df_log['gdp_pro_capite'].max()-df_log['gdp_pro_capite'].min())/60
#creo una lista con gli intervalli delle varie colonne
min_val=df_log['gdp_pro_capite'].min()
max_val=df_log['gdp_pro_capite'].max()
#definisco valori minimi e massimi per comprendere massimo e minimo
start_bin=np.floor(min_val / bin_width) * bin_width
end_bin=np.ceil(max_val / bin_width) * bin_width + bin_width
common_bins=np.arange(start_bin, end_bin, bin_width)
#aggiungo le parti relative alle diverse etichette con colori diversi
for label in labels:
    data = df_log[df_log['GDP_procap_etichetta'] == label]['gdp_pro_capite']
    if len(data)>0:
        plt.hist(data, bins=common_bins, color=labels[label], label=label, edgecolor='black')
plt.title(f'Istogramma del logaritmo di gdp_pro_capite')
plt.xlabel(col)
plt.ylabel('Frequenza')
plt.tight_layout()
plt.savefig(output_path)
plt.close()

#ripeto per la colonna eolico e le etichette windclass
#lista con i colori associati alle varie classi
colors=['gray','yellow','orange','red','green','blue','violet']
output_folder1 = os.path.join(output_folder, "logaritmi")
os.makedirs(output_folder1, exist_ok=True)
output_path = os.path.join(output_folder1, f"eolico_istogramma.png")
plt.figure(figsize=(10, 15))
df_log=df[['eolico','Wind_class']]
df_log=df_log[df_log['eolico'] != 0]
df_log['eolico']=np.log(df_log['eolico'])
#imposto la larghezza delle varie colonne
bin_width=(df_log['eolico'].max()-df_log['eolico'].min())/60
#creo una lista con gli intervalli delle varie colonne
min_val=df_log['eolico'].min()
max_val=df_log['eolico'].max()
#definisco valori minimi e massimi per comprendere massimo e minimo
start_bin=np.floor(min_val / bin_width) * bin_width
end_bin=np.ceil(max_val / bin_width) * bin_width + bin_width
common_bins=np.arange(start_bin, end_bin, bin_width)
for i in range(1,8):
    data = df_log[df_log["Wind_class"] == i]["eolico"]
    if len(data)>0:
        plt.hist(data, bins=common_bins, color=colors[i-1], label=i, edgecolor='black')
plt.title(f'Istogramma del logaritmo di eolico')
plt.xlabel("eolico")
plt.ylabel('Frequenza')
plt.tight_layout()
plt.savefig(output_path)
plt.close()