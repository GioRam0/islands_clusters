import os
import geopandas as gp

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..")

#importo il dataframe
ris_folder = os.path.join(cartella_corrente, "clusters_stat")
os.makedirs(ris_folder, exist_ok=True)

colonne_escludere=['ALL_Uniq', 'Name_USGSO', 'Densità_pop_etichetta', 'Solar_etichetta', 'GDP_procap_etichetta', 'Wind_class', 'NO_res', 'geometry', 'Cluster_Label']

#funzione che crea statistiche per i singoli clusters
def creaz_stat(i):
    gdf_path=os.path.join(cartella_corrente, f"geodataframe_clusterizzati/{i}/isole_clusters.gpkg")
    gdf=gp.read_file(gdf_path)
    colonne_includere=[col for col in gdf.columns if col not in colonne_escludere]
    agg_functions = {col: ['min', 'max', 'mean'] for col in colonne_includere}
    clustered_stats = gdf.groupby('Cluster_Label').agg(agg_functions)
    clustered_stats.columns = ['_'.join(col).strip() for col in clustered_stats.columns.values]
    clustered_stats = clustered_stats.reset_index()
    output_path=os.path.join(ris_folder, f'statistiche_descrittive_{i}_clusters.xlsx')
    clustered_stats.to_excel(output_path, index=False)

for i in range(1,11):
    creaz_stat(i)