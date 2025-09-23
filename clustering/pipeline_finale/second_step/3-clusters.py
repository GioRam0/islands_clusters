import os
import pandas as pd
from sklearn.cluster import KMeans

cartella_corrente = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(cartella_corrente, '../first_step/dataframes/df_norm_first_step.csv')
df = pd.read_csv(csv_path)
#aggiungo le colonne per i clusters finali
df['cluster_finali'] = -1

#possibili colonne su cui effettuare il clustering
colonne = [['solar_pow', 'eolico', 'superficie_res', 'solar_seas_ind', 'eolico_std', 'offshore', 'evi', 'hydro', 'geothermal_potential'],
           ['solar_pow', 'eolico', 'superficie_res', 'solar_seas_ind', 'eolico_std', 'offshore', 'evi', 'hydro'],
           ['solar_pow', 'eolico', 'superficie_res', 'solar_seas_ind', 'eolico_std', 'offshore', 'evi']
]
#iperparametri del kmeans per i diversi cluster (numero di cluster e indice delle colonne da prendere)
iperparam = {
    0 : [3, 0],
    1 : [3, 0],
    2 : [3, 0],
    3 : [3, 1],
    4 : [2, 0],
    5 : [4, 0],
    6 : [3, 0],
    7 : [3, 0],
    8 : [3, 0],
    9 : [3, 0],
    10 : [2, 1],
    11 : [2, 0]
}

#itero per i vari clusters del primo step
for clust, iper in iperparam.items():
    print(f'cluster {clust}')
    df1 = df[df['cluster']==clust].copy()
    #clustering
    kmeans = KMeans(n_clusters=iper[0], init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(df1[colonne[iper[1]]])
    #aggiorno la colonna cluster_finali del dataframe df
    df1['cluster_finali'] = kmeans.labels_
    for i,isl in df1.iterrows():
        df.loc[i, 'cluster_finali'] = df1.loc[i,'cluster_finali']

#creo una colonna con un id univoco dei cluster finali
df["cluster_id"] = df["cluster"].astype(str) + "." + df["cluster_finali"].astype(str)

#esportazione dataframe
output_folder = os.path.join(cartella_corrente, 'results/dataframes')
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'df_norm_final.csv')
df.to_csv(output_path, index=False, encoding='utf-8')

#importo il dataframe raw, aggiungo la colonna clusters_finali ed esporto
csv_path = os.path.join(cartella_corrente, '../first_step/dataframes', 'df_raw_first_step.csv')
df1 = pd.read_csv(csv_path)
df1['cluster_finali'] = df['cluster_finali']
output_folder = os.path.join(cartella_corrente, 'results/dataframes')
output_path = os.path.join(output_folder, 'df_raw_final.csv')
df1.to_csv(output_path, index=False, encoding='utf-8')