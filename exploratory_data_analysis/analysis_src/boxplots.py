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
#colonne numeriche per cui non ha senso realizzare i boxplots
colonne_da_escludere = ['ALL_Uniq', 'Wind_class', 'NO_res']
colonne_da_includere = [col for col in df.columns if col not in colonne_da_escludere]

#creo ed esporto i boxplots
output_folder = os.path.join(ris_folder, "boxplots")
os.makedirs(output_folder, exist_ok=True)
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
    #faccio un nuovo istogramma per le colonne con molti zeri
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