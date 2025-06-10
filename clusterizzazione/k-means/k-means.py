import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import geopandas as gp

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..")

#importo il dataframe
pkl_path = os.path.join(cartella_progetto, "exploratory_data_analisys/risultati/analisys_df.pkl")
df = pd.read_pickle(pkl_path)
colonne_norm=['evi', 'temp', 'prec', 'solar_pow']
colonne_log=['IslandArea', 'Popolazione', 'Densità_pop', 'eolico', 'offshore', 'gdp', 'gdp_pro_capite', 'geothermal_potential', 'hydro', 'urban_area', 'ele_max']
colonne_perc=['urban_area_rel', 'superficie_res']
colonne_escludere=['ALL_Uniq', 'Name_USGSO', 'Densità_pop_etichetta', 'Solar_etichetta', 'GDP_procap_etichetta', 'Wind_class', 'NO_res']
colonne_includere=[col for col in df.columns if col not in colonne_escludere]

#nuovo dataframe
df_normalized = df.copy()
scaler = StandardScaler()
# Applica la normalizzazione normale alle colonne specificate
df_normalized[colonne_norm] = scaler.fit_transform(df[colonne_norm])
for col in colonne_log:
    df_normalized[col] = np.log1p(df_normalized[col])
    df_normalized[col] = scaler.fit_transform(df_normalized[[col]])
for col in colonne_perc:
    df_normalized[col]=(df_normalized[col]-50)/25

gdf_path=os.path.join(cartella_progetto, 'data/isole_filtrate/finali/isole_arro4.gpkg')
gdf=gp.read_file(gdf_path)
output_path=os.path.join(cartella_corrente, 'geodataframe_clusterizzati')
os.makedirs(output_path, exist_ok=True)

wcss = []
for i in range(1, 11):
    df1=df.copy()
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(df_normalized[colonne_includere])
    df1['Cluster_Label'] = kmeans.labels_
    df1 = pd.merge(df1, gdf[['ALL_Uniq', 'geometry']], on='ALL_Uniq', how='left')
    gdf1=gp.GeoDataFrame(df1, geometry='geometry', crs="EPSG:4326")
    folder_out=os.path.join(output_path, f'{i}')
    os.makedirs(folder_out, exist_ok=True)
    out_file=os.path.join(folder_out, 'isole_clusters.gpkg')
    gdf1.to_file(out_file, driver="GPKG")
    for j in range(i):
        gdf2=gdf1[(gdf1['Cluster_Label']==j)]
        out_file=os.path.join(folder_out, f'isole_clusters_{j}.gpkg')
        gdf2.to_file(out_file, driver="GPKG")
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12, 10))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS (Within-cluster sum of squares)')
plt.grid(True)
output_path=os.path.join(cartella_corrente,'elbow_method.png')
plt.savefig(output_path)
plt.close()