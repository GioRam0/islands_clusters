import os
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.metrics import silhouette_score
#importo l'algoritmo di clustering
from active_semi_clustering.semi_supervised.pairwise_constraints import MKMeans

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

#funzione che prende in input le labels dei cluster e calcola quanti vincoli sono stati violati
def calcolo_violazioni(labels):
    cont = 0
    for vincolo in cl:
        if labels[vincolo[0]]==labels[vincolo[1]]:
            cont+=1
    return cont/len(cl)

#ripeto la run diverse volte, salvo i risultati con un numero diverso ogni volta
def funzione(i):
    shilouette = []
    violazioni = []
    #itero per il numero di cluster ed esporto i grafici cluster-silhouette e cluster-violazioni
    for n_clust in range(5,31):
        mk = MKMeans(n_clusters=n_clust, max_iter = 10000)
        mk.fit(df[colonne_includere].values, cl=cl)
        score_mk = silhouette_score(df[colonne_includere], mk.labels_)
        shilouette.append(score_mk)
        df[f'cluster_label_mk_{n_clust}']=mk.labels_
        violazione = calcolo_violazioni(mk.labels_)
        violazioni.append(violazione)

    folder_path = os.path.join(cartella_corrente, 'silhouette')
    os.makedirs(folder_path,exist_ok=True)
    plt.figure(figsize=(12, 10))
    plt.plot(range(5,31), shilouette, marker='o')
    plt.title(f'Elbow Method for Optimal K')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel(f'Shilouette_score MKMeans')
    plt.grid(True)
    output_path=os.path.join(folder_path,f'elbow_method_run_{i}.png')
    plt.savefig(output_path)
    plt.close()
    folder_path = os.path.join(cartella_corrente, 'violazioni')
    os.makedirs(folder_path,exist_ok=True)
    plt.figure(figsize=(12, 10))
    plt.plot(range(5,31), violazioni, marker='o')
    plt.title(f'vincoli violati per cluster')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel(f'Violazioni Totali')
    plt.grid(True)
    output_path=os.path.join(folder_path,f'violazioni_run_{i}.png')
    plt.savefig(output_path)
    plt.close()

    #esportazione dataframe
    folder_path = os.path.join(cartella_corrente, 'dataframes')
    os.makedirs(folder_path,exist_ok=True)
    output_path = os.path.join(folder_path, f'df_etichette_vari_k_run_{i}.csv')
    df.to_csv(output_path)

#ripeto la run diverse volte
for i in range(10):
    print(f'run {i+1}')
    funzione(i)