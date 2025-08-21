#importo le librerie
import os
import pandas as pd
from sklearn.cluster import KMeans

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..")

#importo il dataframe
csv_path = os.path.join(cartella_corrente, 'dataframes', 'df_norm_first_step.csv')
df = pd.read_csv(csv_path)
#aggiungo le colonne per i clusters finali
df['clusters_finali'] = -1

metodi = {
    0 : [2, ['solar_pow', 'eolico', 'superficie_res', 'solar_seas_ind', 'eolico_std', 'offshore', 'evi', 'hydro', 'geothermal_potential']],
    1 : [2, ['solar_pow', 'eolico', 'superficie_res', 'solar_seas_ind', 'eolico_std', 'offshore', 'evi', 'hydro', 'geothermal_potential']],
    2 : [2, ['solar_pow', 'eolico', 'superficie_res', 'solar_seas_ind', 'eolico_std', 'offshore', 'evi', 'hydro', 'geothermal_potential']],
    3 : [1, []],
    4 : [2, ['solar_pow', 'eolico', 'superficie_res', 'solar_seas_ind', 'eolico_std', 'offshore', 'evi', 'hydro', 'geothermal_potential']],
    5 : [3, ['solar_pow', 'eolico', 'superficie_res', 'solar_seas_ind', 'eolico_std', 'offshore', 'evi', 'hydro', 'geothermal_potential']],
    6 : [3, ['solar_pow', 'eolico', 'superficie_res', 'solar_seas_ind', 'eolico_std', 'offshore', 'evi', 'hydro', 'geothermal_potential']],
    7 : [2, ['solar_pow', 'eolico', 'superficie_res', 'solar_seas_ind', 'eolico_std', 'offshore']],
    8 : [2, ['solar_pow', 'eolico', 'superficie_res', 'solar_seas_ind', 'eolico_std', 'offshore', 'evi', 'hydro']],
    9 : [2, ['solar_pow', 'eolico', 'superficie_res', 'solar_seas_ind', 'eolico_std', 'offshore', 'evi', 'hydro', 'geothermal_potential']],
    10 : [3, ['solar_pow', 'eolico', 'superficie_res', 'solar_seas_ind', 'eolico_std', 'offshore', 'evi', 'hydro']],
    11 : [1, []]
}

#itero per i vari clusters
for clust, methods in metodi.items():
    #analisi
    print(f'cluster {clust}:')
    df1 = df[df['clusters']==clust].copy()
    l = len(df1)
    print(f'Numero di isole nel cluster {clust}: {l}')
    l = len(df1[df1['hydro']>0])
    print(f'Isole nel cluster con hydro > 0 : {l}')
    l = len(df1[df1['geothermal_potential']>0])
    print(f'Isole nel cluster con geothermal_potential > 0 : {l}')
    l = len(df1[df1['offshore']>0])
    print(f'Isole nel cluster con offshore > 0 : {l}')
    l = len(df1[df1['Solar_etichetta']=='L'])
    print(f'Isole nel cluster con Solar_etichetta == L : {l}')
    l = len(df1[df1['Solar_etichetta']=='M'])
    print(f'Isole nel cluster con Solar_etichetta == M : {l}')
    l = len(df1[df1['Solar_etichetta']=='S'])
    print(f'Isole nel cluster con Solar_etichetta == S : {l}')
    l = len(df1[df1['Wind_class'] > 5])
    print(f'Isole nel cluster con Wind_class > 5 : {l}')
    l = len(df1[(df1['Wind_class']<6) & (df1['Wind_class']>3)])
    print(f'Isole nel cluster con 3 < Wind_class < 6 : {l}')
    l = len(df1[df1['Wind_class'] < 4])
    print(f'Isole nel cluster con Wind_class < 4 : {l}')
    if methods[0]>1:
        #clusterizzazione
        kmeans = KMeans(n_clusters=methods[0], init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(df1[methods[1]])
        #aggiorno la colonna clusters_finali del dataframe df
        df1['clusters_finali'] = kmeans.labels_
    else:
        df1['clusters_finali'] = 0
    #aggiorno il dataframe iniziale con i clusters finali
    tot = df['clusters_finali'].max() + 1
    for k,(i, isl) in enumerate(df1.iterrows()):
        if isl.ALL_Uniq != df.loc[i, 'ALL_Uniq']:
            print('errore corrispondenze isole')
        df.loc[i, 'clusters_finali'] = tot + isl.clusters_finali

#cluster 0: a grandi linee sottocluster 0 poco vento risorse hydro offshore geotermico, discreto sole, sottocluster 1 poco sole molto vento
#cluster 1: sottocluster 1 poco sole molto vento, meno numeroso
#cluster 2: sottocluster 0 poco sole molto vento, poco numeroso
#cluster 3: scarso vento e sole diffuso, abbastanza hydro, non sottoclusters
#cluster 4: sottocluster 0 molte isole poco vento, sottocluster 1 meno numeroso piu vento poco sole
#cluster 5 (cluster meno definito): sottocluster 0 piu numeroso poco vento poche risorse extra, sottocluster 1 meno numeroso poco sole, discreto vento, sottocluster 2 piu numeroso discreto offshore e sole, poco vento
#cluster 6: sottocluster 0 mediamente numeroso geothermal resto medio, sottocluster 1 piu numeroso buon hydro e offshore, poco vento discreto sole,sottocluster 2 molto vento poco sole poco numeroso
#cluster  7 poche isole con potenziale idro significativo, scelgo variabili fino a offshore (hopkins non ottimali):
#sottocluster 0 con risorse varie non sembra una ottima clusterizzazione, sottocluster 1 poche isole con hydro, offshore e poco sole
#cluster  8 (esclusa geothermal dall'analisi), poco vento i generale: secondo sottocluster poco numeroso poche risorse, primo discreto sole un po di hydro 
#cluster  9 (in teoria 4 clusters con geo, o levo geo e faccio 3 clusters(solo 6 elementi)), mette 4 clusters perche 3 isole hanno moltissimo offshore
#sottocluster 0 molte isole abbastanza varie (prob), sottocluster 1 3 elementi offshore, sottocluster 2 abbastanza elementi poco sole e buon offshore, sottocluster 3 poche isole per geothermal
#cluster  10 (no geothermal): sottocluster 0 molto hydro, sottocluster 1 risorse abbondanti, sottocluster 2 1 elemento a causa di offshore
#cluster  11: poco clusterizzabile, elementi con hydro rilevante con alta densita e consumi

#droppo le colonne inutili
df = df.drop(columns=['cannot', 'must', 'clusters'])
#esportazione
output_folder = os.path.join(cartella_corrente, 'dataframes')
output_path = os.path.join(output_folder, 'df_norm_final.csv')
df.to_csv(output_path)

#importo il dataframe raw, aggiungo la colonna clusters_finali, droppo quelle inutili ed esporto
csv_path = os.path.join(cartella_corrente, 'dataframes', 'df_raw_first_step.csv')
df1 = pd.read_csv(csv_path, index=False, encoding='utf-8')
df1['clusters_finali'] = df['clusters_finali']
df1 = df1.drop(columns=['cannot', 'must', 'clusters', 'clusters_list'])
output_folder = os.path.join(cartella_corrente, 'dataframes')
output_path = os.path.join(output_folder, 'df_raw_final.csv')
df1.to_csv(output_path, index=False, encoding='utf-8')