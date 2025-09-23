import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

cartella_corrente = os.path.dirname(os.path.abspath(__file__))

df_folder = os.path.join(cartella_corrente, "..", "..")
csv_path=os.path.join(df_folder, 'df_raw.csv')
df = pd.read_csv(csv_path)
#colonne numeriche per cui non ha senso realizzare i boxplots
colonne_da_escludere = ['ALL_Uniq', 'Wind_class', 'NO_res']
colonne_da_includere = [col for col in df.columns if col not in colonne_da_escludere]

#creo ed esporto i boxplots
ris_folder = os.path.join(cartella_corrente, "..", "results")
os.makedirs(ris_folder, exist_ok=True)
output_folder = os.path.join(ris_folder, "boxplots")
os.makedirs(output_folder, exist_ok=True)
#itero per le colonne numeriche da includere
for col in df[colonne_da_includere].select_dtypes(include='number').columns:
    output_folder1 = os.path.join(output_folder, "normali")
    os.makedirs(output_folder1, exist_ok=True)
    output_path = os.path.join(output_folder1, f"{col}_boxplot.png")
    plt.figure(figsize=(10, 15))
    sns.boxplot(x=df[col], showfliers=True, color='skyblue')
    plt.title(f"Boxplot di {col}")
    plt.xlabel(col)
    plt.ylabel("Frequenza")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    #faccio un nuovo boxplot solo per le colonne con molti zeri
    conteggio_zeri = (df[col] == 0).sum()
    percentuale_zeri = conteggio_zeri / len(df)
    if percentuale_zeri>0.5:
        df_senza_zeri = df[df[col] != 0]
        plt.figure(figsize=(10, 15))
        sns.boxplot(x=df_senza_zeri[col], showfliers=True, color='green')
        plt.title(f"Boxplot di {col}")
        plt.xlabel(col)
        plt.ylabel("Frequenza")
        plt.tight_layout()
        output_folder1 = os.path.join(output_folder, "no_zeri")
        os.makedirs(output_folder1, exist_ok=True)
        output_path = os.path.join(output_folder1, f"{col}_boxplot.png")
        plt.savefig(output_path)
        plt.close()