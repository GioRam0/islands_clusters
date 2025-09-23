import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

cartella_corrente = os.path.dirname(os.path.abspath(__file__))

#importo il dataframe
df_folder = os.path.join(cartella_corrente, "..", "..")
csv_path=os.path.join(df_folder, 'df_raw.csv')
df = pd.read_csv(csv_path)
#colonne con distribuzioni a coda
colonne_code=['IslandArea', 'Popolazione', 'Densità_pop', 'offshore', 'gdp_2019', 'geothermal_potential', 'hydro', 'urban_area', 'urban_area_rel','ele_max']

#creo ed esporto gli istogrammi
ris_folder = os.path.join(cartella_corrente, "..", "results")
os.makedirs(ris_folder, exist_ok=True)
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

#per consumptions, densita popolazione ed eolico faccio l'isogramma con colori diversi in base all'etichetta
#dizionario con le etichette possibili e i colori che voglio loro associare nell'istogramma
labels={'S': 'green', 'M':'yellow', 'L': 'blue'}
output_folder1 = os.path.join(output_folder, "logaritmi")
os.makedirs(output_folder1, exist_ok=True)
output_path = os.path.join(output_folder1, f"Densità_pop_istogramma.png")
plt.figure(figsize=(10, 15))
df_log=df[['Densità_pop','Densità_pop_etichetta']]
df_log=df_log[df_log['Densità_pop'] != 0]
df_log['Densità_pop']=np.log(df_log['Densità_pop'])
#imposto la larghezza delle varie colonne e creo una lista di intervalli
bin_width=(df_log['Densità_pop'].max()-df_log['Densità_pop'].min())/60
min_val=df_log['Densità_pop'].min()
max_val=df_log['Densità_pop'].max()
start_bin=np.floor(min_val / bin_width) * bin_width
end_bin=np.ceil(max_val / bin_width) * bin_width + bin_width
common_bins=np.arange(start_bin, end_bin, bin_width)
#aggiungo le parti relative alle diverse etichette con colori diversi
for label in labels:
    data = df_log[df_log['Densità_pop_etichetta'] == label]['Densità_pop']
    if len(data)>0:
        plt.hist(data, bins=common_bins, color=labels[label], label=label, edgecolor='black')
plt.title(f'Istogramma del logaritmo di Densità_pop')
plt.xlabel('Densità_pop')
plt.ylabel('Frequenza')
plt.tight_layout()
plt.savefig(output_path)
plt.close()

colors=['gray','yellow','orange','red','green','blue','violet']
output_folder1 = os.path.join(output_folder, "logaritmi")
os.makedirs(output_folder1, exist_ok=True)
output_path = os.path.join(output_folder1, f"eolico_istogramma.png")
plt.figure(figsize=(10, 15))
df_log=df[['eolico','Wind_class']]
df_log=df_log[df_log['eolico'] != 0]
df_log['eolico']=np.log(df_log['eolico'])
bin_width=(df_log['eolico'].max()-df_log['eolico'].min())/60
min_val=df_log['eolico'].min()
max_val=df_log['eolico'].max()
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

labels={'XS': 'red', 'S': 'green', 'M':'yellow', 'L': 'blue'}
output_folder1 = os.path.join(output_folder, "logaritmi")
os.makedirs(output_folder1, exist_ok=True)
output_path = os.path.join(output_folder1, f"consumption_istogramma.png")
plt.figure(figsize=(10, 15))
df_log=df[['consumption','consumption_etichetta']]
df_log=df_log[df_log['consumption'] != 0]
df_log['consumption']=np.log(df_log['consumption'])
bin_width=(df_log['consumption'].max()-df_log['consumption'].min())/60
min_val=df_log['consumption'].min()
max_val=df_log['consumption'].max()
start_bin=np.floor(min_val / bin_width) * bin_width
end_bin=np.ceil(max_val / bin_width) * bin_width + bin_width
common_bins=np.arange(start_bin, end_bin, bin_width)
for label in labels:
    data = df_log[df_log['consumption_etichetta'] == label]['consumption']
    if len(data)>0:
        plt.hist(data, bins=common_bins, color=labels[label], label=label, edgecolor='black')
plt.title(f'Istogramma del logaritmo di consumptions')
plt.xlabel('consumption')
plt.ylabel('Frequenza')
plt.tight_layout()
plt.savefig(output_path)
plt.close()