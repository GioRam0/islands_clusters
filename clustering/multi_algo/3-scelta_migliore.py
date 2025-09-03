#importo le librerie
import numpy as np
import os
import pandas as pd
import json

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..")

#importo il dataframe
pkl_path = os.path.join(cartella_corrente, "results/clustering_results.pkl")
df = pd.read_pickle(pkl_path)

#funzione che estrae il random state dal dizionario delle options se presente
def extract_random_state(d):
    if isinstance(d,dict):
        return d.pop('random_state', np.nan)
    return np.nan
#creo una nuova colonna con il random state estraendolo dal dizionario    
df['random_state'] = df['options'].apply(extract_random_state)

#stessa cosa con il numero di cluster
def extract_n_clusters(d):
    return d.pop('n_clusters')
df['n_clusters'] = df['options'].apply(extract_n_clusters)

df_clean=df.copy()
#creo una stringa identificativa della riga che considera tutto escluso il random state, due elementi che differiscono solo per il random state avranno questa stringa uguale
df_clean['options_str'] = df_clean['options'].apply(lambda x: json.dumps(x, sort_keys=True))
df_clean['config_key'] = df_clean.apply(lambda row: (
    row['algorithm'],
    row['n_clusters'],
    row['options_str']
), axis=1)
#degli elementi con la stessa stringa conservo quello che fa prestazioni migliori nelle varie metriche
best_silhouette = df_clean.loc[df_clean.groupby('config_key')['silhouette_score'].idxmax()]
best_calinski = df_clean.loc[df_clean.groupby('config_key')['calinski_harabasz_score'].idxmax()]
best_davies = df_clean.loc[df_clean.groupby('config_key')['davies_bouldin_score'].idxmin()]

#individuo le combinazioni che performano meglio nelle varie metriche
a=1
common_keys=set()
common_keys1=set()
common_keys2=set()
common_keys3=set()
while True:
    top_silhouette = best_silhouette.sort_values(by='silhouette_score', ascending=False).head(a)
    top_calinski = best_calinski.sort_values(by='calinski_harabasz_score', ascending=False).head(a)
    top_davies = best_davies.sort_values(by='davies_bouldin_score', ascending=True).head(a)
    a+=1
    if len(common_keys)==0:
        common_keys = set(top_silhouette['config_key']) & \
                      set(top_calinski['config_key']) & \
                      set(top_davies['config_key'])
        if len(common_keys)>0:
            print(f"Found {len(common_keys)} common configurations with top scores at a={a-1}:")
            print(common_keys)
    if len(common_keys1)==0:
        common_keys1 = set(top_silhouette['config_key']) & \
                      set(top_calinski['config_key'])
        if len(common_keys1)>0:
            print(f"Found {len(common_keys1)} common configurations in sil and cal with top scores at a={a-1}:")
            print(common_keys1)
    if len(common_keys2)==0:
        common_keys2 = set(top_silhouette['config_key']) & \
                      set(top_davies['config_key'])
        if len(common_keys2)>0:
            print(f"Found {len(common_keys2)} common configurations in sil and dav with top scores at a={a-1}:")
            print(common_keys2)
    if len(common_keys3)==0:
        common_keys3 = set(top_calinski['config_key']) & \
                      set(top_davies['config_key'])
        if len(common_keys3)>0:
            print(f"Found {len(common_keys3)} common configurations in cal and dav with top scores at a={a-1}:")
            print(common_keys3)
    if common_keys and common_keys1 and common_keys2 and common_keys3:
        break

#esportazione
df_common = df_clean[df_clean['config_key'].isin(common_keys)]
df_common1 = df_clean[df_clean['config_key'].isin(common_keys1)]
df_common2 = df_clean[df_clean['config_key'].isin(common_keys2)]
df_common3 = df_clean[df_clean['config_key'].isin(common_keys3)]

if len(df_common)>1:
    print('ok')
    df_common = df_common.loc[df['silhouette_score'].idxmax()]
if len(df_common1)>1:
    df_common1 = df_common1[df_common1['silhouette_score'] == df_common1['silhouette_score'].max()]
if len(df_common2)>1:
    df_common2 = df_common2[df_common2['silhouette_score'] == df_common2['silhouette_score'].max()]
if len(df_common3)>1:
    print('ncjsbchdvbhcdh')
    df_common3 = df_common3[df_common3['calinski_harabasz_score'] == df_common3['silhouette_score'].max()]

df_common.to_pickle(os.path.join(cartella_corrente, "results/best_configs.pkl"))
if df_common1.iloc[0]['config_key'] != df_common.iloc[0]['config_key']:
    df_common1.to_pickle(os.path.join(cartella_corrente, "results/best_configs1.pkl"))
if df_common2.iloc[0]['config_key'] != df_common.iloc[0]['config_key']:
    if df_common2.iloc[0]['config_key'] != df_common1.iloc[0]['config_key']:
        df_common2.to_pickle(os.path.join(cartella_corrente, "results/best_configs2.pkl"))
if df_common3.iloc[0]['config_key'] != df_common.iloc[0]['config_key']:
    if df_common3.iloc[0]['config_key'] != df_common1.iloc[0]['config_key']:
        if df_common3.iloc[0]['config_key'] != df_common2.iloc[0]['config_key']:
            df_common3.to_pickle(os.path.join(cartella_corrente, "results/best_configs3.pkl"))