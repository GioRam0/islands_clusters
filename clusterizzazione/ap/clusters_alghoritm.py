#importo le librerie
import numpy as np
from sklearn.cluster import AffinityPropagation
import pickle
import os
import pandas as pd
import geopandas as gp

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..")

#importo la matrice di similarita
pkl_path = os.path.join(cartella_corrente, "preparazione", "similarity_matrix.pkl")
with open(pkl_path, 'rb') as f:
    similarity_matrix = pickle.load(f)
pkl_path = os.path.join(cartella_corrente, "preparazione", "median_distance.pkl")
with open(pkl_path, 'rb') as f:
    median_distance = pickle.load(f)

#algoritmo affinity propagation
affprop = AffinityPropagation(affinity='precomputed', preference=median_distance*100, random_state=10)
affprop.fit(similarity_matrix)

#numero di clusters ottenuti
clust_numb = len(affprop.cluster_centers_indices_)
print(clust_numb)

#importo il dataframe con i dati normalizzati e creo una colonna con le etichette dei clusters ed esporto
pkl_path = os.path.join(cartella_progetto, "exploratory_data_analysis/normalization/risultati/analysis_df.pkl")
df = pd.read_pickle(pkl_path)
df = df.reset_index(drop=True)
df['Cluster_label']=affprop.labels_
folder_out=os.path.join(cartella_corrente, "risultati")
os.makedirs(folder_out, exist_ok=True)
output_path = os.path.join(folder_out, 'df_norm.pkl')
df.to_pickle(output_path)
#centri dei clusters
centri_indices = affprop.cluster_centers_indices_
df_centri = df.iloc[centri_indices].copy()
folder_out=os.path.join(cartella_corrente, "risultati")
output_path = os.path.join(folder_out, 'centri_norm.pkl')
df_centri.to_pickle(output_path)

#ripeto con i dati originali
pkl_path = os.path.join(cartella_progetto, "exploratory_data_analysis/raw/risultati/analysis_df.pkl")
df = pd.read_pickle(pkl_path)
df = df.reset_index(drop=True)
df['Cluster_label']=affprop.labels_
folder_out=os.path.join(cartella_corrente, "risultati")
output_path = os.path.join(folder_out, 'df_raw.pkl')
df.to_pickle(output_path)
#centri dei clusters
centri_indices = affprop.cluster_centers_indices_
df_centri = df.iloc[centri_indices].copy()
folder_out=os.path.join(cartella_corrente, "risultati")
output_path = os.path.join(folder_out, 'centri_raw.pkl')
df_centri.to_pickle(output_path)

#creiamo ed esportiamo i geodataframe dei singoli clusters per visualizzarli
pkl_path = os.path.join(cartella_progetto, 'data/isole_filtrate/finali/isole_arro4.gpkg')
gdf = gp.read_file(pkl_path)
df = pd.merge(df, gdf[['ALL_Uniq', 'geometry']], on='ALL_Uniq', how='left')
gdf=gp.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
folder_out=os.path.join(cartella_corrente, "risultati/geo")
os.makedirs(folder_out, exist_ok=True)
out_file=os.path.join(folder_out, 'isole_clusters.gpkg')
gdf.to_file(out_file, driver="GPKG")
for j in range(clust_numb):
    gdf1=gdf[(gdf['Cluster_label']==j)]
    out_file=os.path.join(folder_out, f'isole_clusters_{j}.gpkg')
    gdf.to_file(out_file, driver="GPKG")