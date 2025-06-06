#importo le librerie
import pandas as pd
import os

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))

def esporta_statistiche_per_cluster(df, df_centri, feature_cols, label_cols):
    for cluster_id in sorted(df['Cluster_label'].unique()):
        #cluster corrente
        df_cluster = df[df['Cluster_label'] == cluster_id]
        centro = df_centri.loc[df_centri['Cluster_label'] == cluster_id].iloc[0]

        #colonne numeriche
        numeriche = df_cluster[feature_cols]
        centro_numeriche = centro[feature_cols]
        #dataframe statistiche
        stats = pd.DataFrame({
            'Centro': centro_numeriche,
            'Media': numeriche.mean(),
            'Mediana': numeriche.median(),
            'Max': numeriche.max(),
            'Min': numeriche.min()
        }).T

        #colonne etichette
        etichette_df = pd.DataFrame(columns=label_cols)
        #etichetta centro
        etichette_df.loc['Centro'] = centro[label_cols]
        #lista etichette varie isole
        lista_etichette = {
            col: ', '.join(sorted(df_cluster[col].astype(str).unique()))
            for col in label_cols
        }
        etichette_df.loc['Tutte le etichette'] = lista_etichette

        #esportazione
        filename = os.path.join(output_folder, f'cluster_{cluster_id}.xlsx')
        with pd.ExcelWriter(filename) as writer:
            stats.to_excel(writer, sheet_name='Numeriche')
            etichette_df.to_excel(writer, sheet_name='Etichette')

#importo il dataframe generico e dei centri con dati raw
pkl_path = os.path.join(cartella_corrente, "risultati", "df_raw.pkl")
df = pd.read_pickle(pkl_path)
pkl_path = os.path.join(cartella_corrente, "risultati", "centri_raw.pkl")
df_centri = pd.read_pickle(pkl_path)
#applico la funzione
output_folder = os.path.join(cartella_corrente, 'risultati', 'analisi_risultati', 'raw')
os.makedirs(output_folder, exist_ok=True)
#suddivisione colonne
colonne_da_escludere= ['ALL_Uniq', 'Name_USGSO', 'Cluster_label']
colonne_etichette = ['Densità_pop_etichetta', 'Solar_etichetta', 'GDP_procap_etichetta', 'Wind_class', 'NO_res']
colonne_numeriche = [col for col in df.columns if (col not in colonne_da_escludere and col not in colonne_etichette)]
esporta_statistiche_per_cluster(df,df_centri, colonne_numeriche, colonne_etichette)


#importo il dataframe generico e dei centri con dati norm
pkl_path = os.path.join(cartella_corrente, "risultati", "df_norm.pkl")
df = pd.read_pickle(pkl_path)
pkl_path = os.path.join(cartella_corrente, "risultati", "centri_norm.pkl")
df_centri = pd.read_pickle(pkl_path)
#applico la funzione
output_folder = os.path.join(cartella_corrente, 'risultati', 'analisi_risultati', 'norm')
os.makedirs(output_folder, exist_ok=True)
#suddivisione colonne
colonne_da_escludere= ['ALL_Uniq', 'Name_USGSO', 'Cluster_label']
colonne_etichette = ['Densità_pop_etichetta', 'Solar_etichetta', 'GDP_procap_etichetta', 'Wind_class', 'NO_res']
colonne_numeriche = [col for col in df.columns if (col not in colonne_da_escludere and col not in colonne_etichette)]
esporta_statistiche_per_cluster(df,df_centri, colonne_numeriche, colonne_etichette)