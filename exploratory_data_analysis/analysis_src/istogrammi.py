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
#colonne numeriche per cui non serve fare gli istogrammi
colonne_da_escludere = ['ALL_Uniq', 'Wind_class', 'NO_res']
colonne_da_includere = [col for col in df.columns if col not in colonne_da_escludere]
#colonne con etichetta e relativa colonna etichetta per cui costruisco istogrammi in modo diverso
colonne_con_etichetta={"Densità_pop":"Densità_pop_etichetta","solar_pow":"Solar_etichetta","gdp_pro_capite":"GDP_procap_etichetta"}
colonne_no_etichetta=[col for col in colonne_da_includere if (col not in colonne_con_etichetta and col != "eolico")]

#creo ed esporto gli istogrammi
output_folder = os.path.join(ris_folder, "istogrammi")
os.makedirs(output_folder, exist_ok=True)
for col in df[colonne_no_etichetta].select_dtypes(include='number').columns:
    output_folder1 = os.path.join(output_folder, "normali")
    os.makedirs(output_folder1, exist_ok=True)
    output_path = os.path.join(output_folder1, f"{col}_istogramma.png")
    plt.figure(figsize=(10, 15))
    df[col].hist(bins=60, color='skyblue', edgecolor='black')
    plt.title(f"Istogramma di {col}")
    plt.xlabel(col)
    plt.ylabel("Frequenza")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    #faccio un nuovo istogramma per le colonne con molti zeri
    conteggio_zeri = (df[col] == 0).sum()
    percentuale_zeri = conteggio_zeri / len(df)
    if percentuale_zeri>0.5:
        df_senza_zeri = df[df[col] != 0]
        plt.figure(figsize=(10, 15))
        plt.hist(df_senza_zeri[col], bins=60, color='green', edgecolor='black')
        plt.title(f'Istogramma di {col} (escludendo gli zeri)')
        plt.xlabel(col)
        plt.ylabel('Frequenza')
        plt.tight_layout()
        output_folder1 = os.path.join(output_folder, "no_zeri")
        os.makedirs(output_folder1, exist_ok=True)
        output_path = os.path.join(output_folder1, f"{col}_istogramma.png")
        plt.savefig(output_path)
        plt.close()

#dizionario con le etichette possibili e i colori che voglio loro associare nell'istogramma
labels={'XS': 'red', 'S': 'green', 'M':'yellow', 'L': 'blue'}
for col in colonne_con_etichetta:
    #etichetta associata alla colonna
    colonna_etichetta=colonne_con_etichetta[col]
    output_folder1 = os.path.join(output_folder, "normali")
    os.makedirs(output_folder1, exist_ok=True)
    output_path = os.path.join(output_folder1, f"{col}_istogramma.png")
    plt.figure(figsize=(10, 15))
    #imposto la larghezza delle varie colonne
    bin_width=(df[col].max()-df[col].min())/60
    #creo una lista con gli intervalli delle varie colonne
    min_val=df[col].min()
    max_val=df[col].max()
    #definisco valori minimi e massimi per comprendere massimo e minimo
    start_bin=np.floor(min_val / bin_width) * bin_width
    end_bin=np.ceil(max_val / bin_width) * bin_width + bin_width
    common_bins=np.arange(start_bin, end_bin, bin_width)
    #aggiungo le parti relative alle diverse etichette con colori diversi
    for label in labels:
        data = df[df[colonna_etichetta] == label][col]
        if len(data)>0:
            plt.hist(data, bins=common_bins, color=labels[label], label=label, edgecolor='black')
    plt.title(f'Istogramma di {col}')
    plt.xlabel(col)
    plt.ylabel('Frequenza')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    #faccio un nuovo istogramma per le colonne con molti zeri
    conteggio_zeri = (df[col] == 0).sum()
    percentuale_zeri = conteggio_zeri / len(df)
    if percentuale_zeri>0.5:
        df_senza_zeri = df[df[col] != 0]
        plt.figure(figsize=(10, 15))
        for label in labels:
            data = df_senza_zeri[df_senza_zeri[colonna_etichetta] == label][col]
            if len(data)>0:
                plt.hist(data, bins=common_bins, color=labels[label], label=label, edgecolor='black')
        plt.title(f'Istogramma di {col} (escludendo gli zeri)')
        plt.xlabel(col)
        plt.ylabel('Frequenza')
        plt.tight_layout()
        output_folder1 = os.path.join(output_folder, "no_zeri")
        os.makedirs(output_folder1, exist_ok=True)
        output_path = os.path.join(output_folder1, f"{col}_istogramma.png")
        plt.savefig(output_path)
        plt.close()

#ripeto per la colonna eolico e le etichette windclass
#lista con i colori associati alle varie classi
colors=['gray','yellow','orange','red','green','blue','violet']
output_folder1 = os.path.join(output_folder, "normali")
os.makedirs(output_folder1, exist_ok=True)
output_path = os.path.join(output_folder1, f"eolico_istogramma.png")
plt.figure(figsize=(10, 15))
#imposto la larghezza delle varie colonne
bin_width=(df['eolico'].max()-df['eolico'].min())/60
#creo una lista con gli intervalli delle varie colonne
min_val=df['eolico'].min()
max_val=df['eolico'].max()
#definisco valori minimi e massimi per comprendere massimo e minimo
start_bin=np.floor(min_val / bin_width) * bin_width
end_bin=np.ceil(max_val / bin_width) * bin_width + bin_width
common_bins=np.arange(start_bin, end_bin, bin_width)
for i in range(1,8):
    data = df[df["Wind_class"] == i]["eolico"]
    if len(data)>0:
        plt.hist(data, bins=common_bins, color=colors[i-1], label=i, edgecolor='black')
plt.title(f'Istogramma di eolico')
plt.xlabel("eolico")
plt.ylabel('Frequenza')
plt.tight_layout()
plt.savefig(output_path)
plt.close()
#faccio un nuovo istogramma per le colonne con molti zeri
conteggio_zeri = (df["eolico"] == 0).sum()
percentuale_zeri = conteggio_zeri / len(df)
if percentuale_zeri>0.5:
    df_senza_zeri = df[df["eolico"] != 0]
    plt.figure(figsize=(10, 15))
    for i in range(1,8):
        data = df_senza_zeri[df_senza_zeri["Wind_class"] == i]['eolico']
        plt.hist(data, bins=common_bins, color=colors[label], label=i, edgecolor='black')
    plt.title(f'Istogramma di eolico (escludendo gli zeri)')
    plt.xlabel('eolico')
    plt.ylabel('Frequenza')
    plt.tight_layout()
    output_folder1 = os.path.join(output_folder, "no_zeri")
    os.makedirs(output_folder1, exist_ok=True)
    output_path = os.path.join(output_folder1, f"eolico_istogramma.png")
    plt.savefig(output_path)
    plt.close()