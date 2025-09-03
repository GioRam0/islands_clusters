#importo le librerie
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))

#importo il dataframe
csv_path=os.path.join(cartella_corrente, 'dataframes', 'df_norm_first_step.csv')
df = pd.read_pickle(csv_path)

#colonne numeriche per cui non ha senso realizzare i KDE-plots
colonne_da_includere = ['solar_pow', 'solar_seas_ind', 'eolico', 'eolico_std', 'offshore', 'evi', 'geothermal_potential', 'hydro', 'superficie_res']
colonne_zeri = ['offshore', 'hydro', 'geothermal_potential']

output_folder = os.path.join(cartella_corrente, "kdeplots")
os.makedirs(output_folder, exist_ok=True)
#definisco una funzione che crea ed esporta i kdeplot
def create_kdeplot(dataframe, number):
    cluster_output_folder = os.path.join(output_folder, f"cluster_{number}")
    os.makedirs(cluster_output_folder, exist_ok=True)
    output_folder1 = os.path.join(cluster_output_folder, "normali")
    os.makedirs(output_folder1, exist_ok=True)
    output_folder2 = os.path.join(cluster_output_folder, "no_zeri")
    os.makedirs(output_folder2, exist_ok=True)
    for col in colonne_da_includere:
        df_cluster = dataframe[dataframe['clusters'] == number]
        output_path = os.path.join(output_folder1, f"{col}_kdeplot.png")
        plt.figure(figsize=(10, 15))
        sns.kdeplot(df_cluster[col], shade=True, color="skyblue", fill=True)
        plt.title(f"KDE Plot di {col}")
        plt.xlabel(col)
        plt.ylabel("Densità")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        #faccio un nuovo istogramma per le colonne con molti zeri
        if col in colonne_zeri:
            df_senza_zeri = df_cluster[df_cluster[col] != 0]
            output_path = os.path.join(output_folder2, f"{col}_kdeplot.png")
            plt.figure(figsize=(10, 15))
            sns.kdeplot(df_senza_zeri[col], shade=True, color="green", fill=True)
            plt.title(f"KDE Plot di {col}")
            plt.xlabel(col)
            plt.ylabel("Densità")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()

for i in range(max(df['clusters'])+1):
    print(f'creating kdeplots for cluster {i}')
    create_kdeplot(df, i)