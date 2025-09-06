#importo le librerie
import os
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.metrics import silhouette_score
import sys
#importo algoritmi di clustering
from active_semi_clustering.semi_supervised.pairwise_constraints import COPKMeans
#mi sa qua non serve
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

shilouette = []
clust_list = []
for n_clust in range(5,26):
    copk = COPKMeans(n_clusters=n_clust)
    try:
        copk.fit(df[colonne_includere].values, cl=cl)
        score_copk = silhouette_score(df[colonne_includere], copk.labels_)
        shilouette.append(score_copk)
        df[f'cluster_label_copk_{n_clust}']=copk.labels_
        clust_list.append(n_clust)
        print(f'Soluzione trovata con {n_clust} cluster')
    except:
        print(f'Soluzione non trovata con {n_clust} cluster')
        
#costruzione grafico silhouette      
plt.figure(figsize=(12, 10))
plt.plot(clust_list, shilouette, marker='o')
plt.title(f'Elbow Method for Optimal K')
plt.xlabel('Number of clusters (k)')
plt.ylabel(f'Shilouette_score COPKMeans')
plt.grid(True)
output_path=os.path.join(cartella_corrente,f'elbow_method.png')
plt.savefig(output_path)
plt.close()

#esportazione
output_path = os.path.join(cartella_corrente, 'df_etichette_vari_k.csv')
df.to_csv(output_path)