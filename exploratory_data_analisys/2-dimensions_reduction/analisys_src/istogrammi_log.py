import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

cartella_corrente = os.path.dirname(os.path.abspath(__file__))

df_folder = os.path.join(cartella_corrente, "..", "..")
csv_path=os.path.join(df_folder, 'df_dim_reduction.csv')
df = pd.read_csv(csv_path)
colonne_code=['superficie_res', 'offshore', 'hydro', 'geothermal_potential', 'gdp_cons_pop_urban_merged', 'hydro', 'solar_seas_ind']

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
    min=df[col].min()
    data=np.log1p(df[col]-min)
    #colonne con molti zeri
    if col=='geothermal_potential' or col=='offshore' or col=='hydro':
        data=np.log1p(df[df[col]>0][col])
    data.hist(bins=60, color='skyblue', edgecolor='black')
    plt.title(f"Istogramma di logaritmo di {col}")
    plt.xlabel(col)
    plt.ylabel("Frequenza")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

#per densita popolazione ed eolico faccio l'isogramma con colori diversi in base all'etichetta
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