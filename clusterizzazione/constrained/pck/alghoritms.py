#importo le librerie
import os
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
pkl_path = os.path.join(cartella_progetto, "exploratory_data_analisys/normalization/risultati/analisys_df.pkl")
df = pd.read_pickle(pkl_path)
colonne_escludere=['ALL_Uniq', 'Name_USGSO', 'Densità_pop_etichetta', 'Solar_etichetta', 'GDP_procap_etichetta', 'Wind_class', 'NO_res']
colonne_includere=[col for col in df.columns if col not in colonne_escludere]

#importo le liste di vincoli
pkl_path = os.path.join(cartella_corrente, '../vincoli', 'must_link.pkl')
ml = pickle.load(open(pkl_path, 'rb'))
pkl_path = os.path.join(cartella_corrente, '../vincoli', 'cannot_link.pkl')
cl = pickle.load(open(pkl_path, 'rb'))
ml=ml[0]+ml[1]+ml[2]+ml[3]+ml[4]+ml[5]
cl=cl[0]+cl[1]+cl[2]+cl[3]+cl[4]+cl[5]

cluster_range = range(5, 20)
shilouette = []

#itero per numeri di clusters diversi
for k in cluster_range:
    #w in pck=, peso violazione vincoli
    #PCKMeans
    print(k)
    pck = PCKMeans(n_clusters=k)
    pck.fit(df[colonne_includere].values, ml=ml, cl=cl)
    score_pck = silhouette_score(df[colonne_includere], pck.labels_)
    df[f'cluster_label_pck_{k}']=pck.labels_
    shilouette.append(score_pck)
    
for k, score in cluster_range:
    print(f"K={k}: {shilouette[k]}")
plt.figure(figsize=(12, 10))
plt.plot(cluster_range, shilouette, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of clusters (k)')
plt.ylabel(f'Shilouette_score PCKMeans')
plt.grid(True)
output_path=os.path.join(cartella_corrente,f'elbow_method.png')
plt.savefig(output_path)
plt.close()

#esportazione
output_path = os.path.join(cartella_corrente, 'df_etichette_vari_k.pkl')
df.to_pickle(output_path)