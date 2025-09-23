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

#itero per i cluster del primo step ed esporto testi con descrizioni etichette
txt_folder = os.path.join(output_folder, 'txt_folder')
os.makedirs(txt_folder, exist_ok=True)
clust_first = df['cluster'].max()+1
for clust in range(clust_first):
    df1 = df[df['cluster']==clust].copy()
    clust_second = df1['cluster_finali'].max()+1
    #file di testo in cui scrivere le statistiche
    txt_path = os.path.join(txt_folder, f'cluster_{clust}.txt')
    with open(txt_path, 'w') as file:
        l = len(df1)
        file.write(f'Numero di isole nel cluster: {l}\n')
        l = len(df1[df1['hydro']>0])
        file.write(f'Isole nel cluster con hydro > 0 : {l}\n')
        l = len(df1[df1['geothermal_potential']>0])
        file.write(f'Isole nel cluster con geothermal_potential > 0 : {l}\n')
        l = len(df1[df1['offshore']>0])
        file.write(f'Isole nel cluster con offshore > 0 : {l}\n')
        l = len(df1[df1['Solar_etichetta']=='L'])
        file.write(f'Isole nel cluster con Solar_etichetta == L : {l}\n')
        l = len(df1[df1['Solar_etichetta']=='M'])
        file.write(f'Isole nel cluster con Solar_etichetta == M : {l}\n')
        l = len(df1[df1['Solar_etichetta']=='S'])
        file.write(f'Isole nel cluster con Solar_etichetta == S : {l}\n')
        l = len(df1[df1['Wind_class'] > 4])
        file.write(f'Isole nel cluster con Wind_class > 4 : {l}\n')
        l = len(df1[(df1['Wind_class']<5) & (df1['Wind_class']>2)])
        file.write(f'Isole nel cluster con 2 < Wind_class < 5 : {l}\n')
        l = len(df1[df1['Wind_class'] < 3])
        file.write(f'Isole nel cluster con Wind_class < 3 : {l}\n')
        file.write('\n')
        for j in range(clust_second):
            df2 = df1[df1['cluster_finali']==j].copy()
            l = len(df2)
            file.write(f'Numero di isole nel sottocluster {j}: {l}\n')
            l = len(df2[df2['hydro']>0])
            file.write(f'Isole nel sottocluster con hydro > 0 : {l}\n')
            l = len(df2[df2['geothermal_potential']>0])
            file.write(f'Isole nel sottocluster con geothermal_potential > 0 : {l}\n')
            l = len(df2[df2['offshore']>0])
            file.write(f'Isole nel sottocluster con offshore > 0 : {l}\n')
            l = len(df2[df2['Solar_etichetta']=='L'])
            file.write(f'Isole nel sottocluster con Solar_etichetta == L : {l}\n')
            l = len(df2[df2['Solar_etichetta']=='M'])
            file.write(f'Isole nel sottocluster con Solar_etichetta == M : {l}\n')
            l = len(df2[df2['Solar_etichetta']=='S'])
            file.write(f'Isole nel sottocluster con Solar_etichetta == S : {l}\n')
            l = len(df2[df2['Wind_class'] > 4])
            file.write(f'Isole nel sottocluster con Wind_class > 4 : {l}\n')
            l = len(df2[(df2['Wind_class']<5) & (df2['Wind_class']>2)])
            file.write(f'Isole nel sottocluster con 2 < Wind_class < 5 : {l}\n')
            l = len(df2[df2['Wind_class'] < 3])
            file.write(f'Isole nel sottocluster con Wind_class < 3 : {l}\n')
            file.write('\n')
        #calcolo la varianza spiegata dal clustering e aggiunta sul .txt
        X = df1[colonne]
        grand_mean = X.mean()
        SST = ((X - grand_mean) ** 2).to_numpy().sum()
        SSB = 0
        for cluster, group in df1[colonne+['cluster_finali']].groupby('cluster_finali'):
            n_k = len(group)
            cluster_mean = group.drop(columns=['cluster_finali']).mean()
            SSB += n_k * ((cluster_mean - grand_mean) ** 2).to_numpy().sum()
        R2 = SSB / SST
        file.write(f'% di varianza nel cluster spiegata dai sottocluster: {R2}')

#folder boxplot
boxplot_folder = os.path.join(output_folder, 'boxplot')
os.makedirs(boxplot_folder, exist_ok=True)
#funzione esportazione boxplot
def box(dataframe, folder_name):
    boxplot_folder1 = os.path.join(boxplot_folder, folder_name)
    os.makedirs(boxplot_folder1, exist_ok=True)
    clust_first = dataframe['cluster'].max()+1
    for clust in range(clust_first):
        df1 = dataframe[dataframe['cluster']==clust].copy()
        clust_second = df1['cluster_finali'].max()+1
        boxplot_folder2 = os.path.join(boxplot_folder1, f'cluster_{clust}')
        os.makedirs(boxplot_folder2, exist_ok=True)
        for feature in colonne:
            plt.figure(figsize=(8, 6))
            data = [df1[df1['cluster_finali'] == cluster_label][feature] for cluster_label in range(clust_second)]
            plt.boxplot(data, vert=True, patch_artist=True)
            plt.xticks(ticks=range(1, clust_second+1), labels=[f'Cluster {label}' for label in range(clust_second)])
            plt.tight_layout()
            boxplot_path = os.path.join(boxplot_folder2, f'{feature}_boxplot.png')
            plt.savefig(boxplot_path)
            plt.close()
box(df, 'normalized')

#folder statistiche descrittive
stat_folder = os.path.join(output_folder, 'descriptive_stat')
os.makedirs(stat_folder, exist_ok=True)
#funzione esportazione statistiche descrittive
def stat(dataframe, folder_name):
    stat_folder1 = os.path.join(stat_folder, folder_name)
    os.makedirs(stat_folder1, exist_ok=True)
    clust_first = dataframe['cluster'].max()+1
    for clust in range(clust_first):
        df1 = dataframe[dataframe['cluster']==clust].copy()
        #excel con statistiche dei singoli cluster
        output_xlsx = os.path.join(stat_folder1, f'statistics_cluster_{clust}.xlsx')
        with pd.ExcelWriter(output_xlsx, engine='xlsxwriter') as writer:
            for feature in colonne:
                stats = []
                clust_second = df1['cluster_finali'].max()+1
                for cluster_label in range(clust_second):
                    df_feature= df1[df1['cluster_finali'] == cluster_label][feature]
                    stats.append([
                        len(df_feature),
                        df_feature.mean(),
                        df_feature.min(),
                        df_feature.quantile(0.2),
                        df_feature.quantile(0.4),
                        df_feature.quantile(0.6),
                        df_feature.quantile(0.8),
                        df_feature.max(),
                        df_feature.std(),
                    ])
                stats_df = pd.DataFrame(
                    np.array(stats).T,
                    index=['len','mean', 'min', '20%', '40%', '60%', '80%', 'max', 'std'],
                    columns=[f'Cluster_{label}' for label in range(clust_second)]
                )
                stats_df.to_excel(writer, sheet_name=feature)
stat(df, 'normalized')

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
        cbar_ticks = np.arange(clust_second) + 0.5
        plt.colorbar(scatter, ticks=cbar_ticks, label='Cluster')
        plt.tight_layout()
        output_path = os.path.join(pca_folder1, f'plot_cluster_{clust}.png')
        plt.savefig(output_path)
        plt.close()
pca(df, 'normalized')

#importo il dataframe raw e applico le funzioni
csv_path = os.path.join(cartella_corrente, 'results/dataframes/df_raw_final.csv')
df = pd.read_csv(csv_path)
colonne = ['evi', 'eolico', 'eolico_std', 'offshore', 'geothermal_potential', 'hydro', 'temp', 'prec', 'hdd', 'cdd', 'solar_pow', 'solar_seas_ind', 'superficie_res']
box(df, 'raw')
stat(df, 'raw')
pca(df, 'raw')