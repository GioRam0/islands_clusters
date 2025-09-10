#importo le librerie
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.metrics import silhouette_score
import sys
#importo algoritmi di clustering
from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans
sys.setrecursionlimit(2000)

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..", "..")

#importo il dataframe
csv_path = os.path.join(cartella_progetto, "exploratory_data_analisys/df_norm.csv")
df = pd.read_csv(csv_path)
colonne_escludere=['ALL_Uniq', 'Name_USGSO', 'Densità_pop_etichetta', 'Solar_etichetta', 'consumption_etichetta', 'Wind_class', 'NO_res']
colonne_includere=[col for col in df.columns if col not in colonne_escludere]

#importo la lista di vincoli
pkl_path = os.path.join(cartella_corrente, '..', '0-constraints', 'cannot_link.pkl')
cl = pickle.load(open(pkl_path, 'rb'))

weights=[0.5,1,2,3]
violazioni = []

#funzione che prende in input le labels dei cluster e calcola quanti vincoli sono stati violati
def calcolo_violazioni(labels):
    cont = 0
    for vincolo in cl:
        if labels[vincolo[0]]==labels[vincolo[1]]:
            cont+=1
    return cont/len(cl)

#itero per numeri di clusters e pesi diversi della violazione
for w in weights:
    shilouette = []
    for n_clust in range(5,26):
        print(f"tentativo con {n_clust} cluster e weight {w}")
        pck = PCKMeans(n_clusters=n_clust, w=w)
        pck.fit(df[colonne_includere].values, cl=cl)
        score_pck = silhouette_score(df[colonne_includere], pck.labels_)
        shilouette.append(score_pck)
        df[f'cluster_label_pck_{n_clust}_{w}']=pck.labels_
        violazione = calcolo_violazioni(pck.labels_)
        if w==0.5:
            violazioni.append([violazione])
        else:
            violazioni[n_clust-5].append(violazione)
    plt.figure(figsize=(12, 10))
    plt.plot(range(5,26), shilouette, marker='o')
    plt.title(f'Elbow Method for Optimal K, w={w}')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel(f'Shilouette_score PCKMeans')
    plt.grid(True)
    folder_out = os.path.join(cartella_corrente, 'Silhouette')
    os.makedirs(folder_out, exist_ok=True)
    output_path=os.path.join(folder_out,f'elbow_method_w_{w}.png')
    plt.savefig(output_path)
    plt.close()

for n in range(len(violazioni)):
    plt.figure(figsize=(12, 10))
    plt.plot(weights, violazioni[n], marker='o')
    plt.title(f'vincoli violati per {n+5} cluster')
    plt.xlabel('violation_weight (w)')
    plt.ylabel(f'violazioni totali')
    plt.grid(True)
    folder_out = os.path.join(cartella_corrente, 'Violations')
    os.makedirs(folder_out, exist_ok=True)
    output_path=os.path.join(folder_out,f'violazioni_n_{n+5}.png')
    plt.savefig(output_path)
    plt.close()

#esportazione
output_path = os.path.join(cartella_corrente, 'df_etichette_vari_k_w.csv')
df.to_csv(output_path)