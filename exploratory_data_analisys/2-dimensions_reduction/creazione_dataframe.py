#importo le librerie
import pandas as pd
import pickle
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))

#importo il dataframe
folder_path = os.path.join(cartella_corrente, "../1-raw/risultati")
pkl_path = os.path.join(folder_path, "analisys_df.pkl")
df = pd.read_pickle(pkl_path)

# unisco 4 variabili con alte correlazioni reciproche
X=df[['gdp','Popolazione', 'urban_area', 'consumption']]
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)
print(f'varianza spiegata dalla prima componente {pca.explained_variance_ratio_}')
df['gdp_cons_pop_urban_merged']=X_pca
#varianza spiegata dalla prima componente [0.83030405]

#considero superficie e superficie res in unico parametro
df['superficie_res']=(df['superficie_res']/100)*df['IslandArea']
#colonne non rilevanti per la clusterizzazione
df=df.drop(columns=['hdd','cdd','prec','ele_max', 'gdp','gdp_2019','consumption','Popolazione', 'urban_area', 'urban_area_rel','IslandArea'])

#esportazione
output_folder = os.path.join(cartella_corrente, 'risultati')
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'analisys_df.pkl')
df.to_pickle(output_path)