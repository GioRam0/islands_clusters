#importo le librerie
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))

#importo il dataframe normalizzato
csv_path = os.path.join(cartella_corrente, 'results/dataframes/df_norm_final.csv')
df = pd.read_csv(csv_path)
colonne = ['solar_pow', 'eolico', 'superficie_res', 'solar_seas_ind', 'eolico_std', 'offshore', 'evi', 'hydro', 'geothermal_potential']

#creazione folder esportazione
output_folder = os.path.join(cartella_corrente, f'results')
os.makedirs(output_folder, exist_ok=True)

#folder PCA
pca_folder = os.path.join(output_folder, 'PCA')
os.makedirs(pca_folder, exist_ok=True)
#funzione esportazione proiezioni PCA
def pca(dataframe, folder_name):
    pca_folder1 = os.path.join(pca_folder, folder_name)
    os.makedirs(pca_folder1, exist_ok=True)
    clust_first = dataframe['cluster'].max()+1
    #proiezione in due dimensioni
    pca = PCA(n_components=2)
    for clust in range(clust_first):
        df1 = dataframe[dataframe['cluster']==clust].copy()
        X_pca = pca.fit_transform(df1[colonne].values)
        df1['PCA1'] = X_pca[:, 0]
        df1['PCA2'] = X_pca[:, 1]
        plt.figure(figsize=(8, 6))
        clust_second = df1['cluster_finali'].max()+1
        cmap = ListedColormap(plt.get_cmap('tab10').colors[:clust_second])
        scatter = plt.scatter(df1['PCA1'], df1['PCA2'], c=df1['cluster_finali'], cmap=cmap, alpha=0.7)
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        cbar_ticks = np.linspace((clust_second-1)/(2*clust_second), (clust_second-1)-((clust_second-1)/(2*clust_second)), clust_second)
        cbar = plt.colorbar(scatter, label='Cluster')
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(range(clust_second))
        plt.tight_layout()
        output_path = os.path.join(pca_folder1, f'plot_cluster_{clust}.png')
        plt.savefig(output_path)
        plt.close()
pca(df, 'normalized')

#importo il dataframe raw e applico le funzioni
csv_path = os.path.join(cartella_corrente, 'results/dataframes/df_raw_final.csv')
df = pd.read_csv(csv_path)
colonne = ['evi', 'eolico', 'eolico_std', 'offshore', 'geothermal_potential', 'hydro', 'temp', 'prec', 'hdd', 'cdd', 'solar_pow', 'solar_seas_ind', 'superficie_res']
pca(df, 'raw')