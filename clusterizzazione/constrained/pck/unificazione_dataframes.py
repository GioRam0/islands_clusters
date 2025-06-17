#importo le librerie
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.metrics import silhouette_score

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..", "..")


#importo il dataframe
pkl_path = os.path.join(cartella_progetto, "exploratory_data_analisys/normalization/risultati/analisys_df.pkl")
df = pd.read_pickle(pkl_path)
colonne_escludere=['ALL_Uniq', 'Name_USGSO', 'Densità_pop_etichetta', 'Solar_etichetta', 'GDP_procap_etichetta', 'Wind_class', 'NO_res', 'eolico_std','solar_seas_ind','temp']
colonne_includere=[col for col in df.columns if col not in colonne_escludere]
pkl_path = os.path.join(cartella_corrente, "df_etichette_vari_k_w_1.pkl")
df = pd.read_pickle(pkl_path)

#importo le liste di vincoli
pkl_path = os.path.join(cartella_corrente, '../vincoli', 'must_link.pkl')
ml = pickle.load(open(pkl_path, 'rb'))
pkl_path = os.path.join(cartella_corrente, '../vincoli', 'cannot_link.pkl')
cl = pickle.load(open(pkl_path, 'rb'))
ml=ml[0]+ml[1]+ml[2]+ml[3]+ml[4]+ml[5]
cl=cl[0]+cl[1]+cl[2]+cl[3]+cl[4]+cl[5]

cluster_range = range(5, 26)
weights_range=np.arange(1,3,0.5)

#funzione per unificare i dataframes, definizione e applicazione
def aggiunta_colonne(dataframe,dataframe1):
    #itero per numeri di clusters e pesi diversi della violazione
    for w in weights_range:
        for k in cluster_range:
            stringa=f'cluster_label_pck_{k}_{w}'
            if stringa in df1.columns:
                dataframe[stringa]=dataframe1[stringa]
for i in range(2,7):
    pkl_path = os.path.join(cartella_corrente, f"df_etichette_vari_k_w_{i}.pkl")
    df1 = pd.read_pickle(pkl_path)
    aggiunta_colonne(df,df1)

#lista da riempire con liste dei vari shilouette scores
shilouette = []
for w in weights_range:
    shil_w=[]
    for k in cluster_range:
        score_pck = silhouette_score(df[colonne_includere], df[f'cluster_label_pck_{k}_{w}'])
        shil_w.append(score_pck)
    shilouette.append(shil_w)

for cont,w in enumerate(weights_range,0):
    for cont1,k in enumerate(cluster_range,0):
        print(f"w={w}, K={k}: {shilouette[cont][cont1]}")
    plt.figure(figsize=(12, 10))
    plt.plot(cluster_range, shilouette[cont], marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel(f'Shilouette_score PCKMeans')
    plt.grid(True)
    output_path=os.path.join(cartella_corrente,f'elbow_method_weight_{w}.png')
    plt.savefig(output_path)
    plt.close()

#esportazione
output_path = os.path.join(cartella_corrente, f'df_etichette_vari_k_w.pkl')
df.to_pickle(output_path)