#importo le librerie
import pandas as pd
import pickle
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))

#importo il dataframe
folder_path = os.path.join(cartella_corrente, "../raw/risultati")
pkl_path = os.path.join(folder_path, "analysis_df.pkl")
df = pd.read_pickle(pkl_path)

# X: dataframe con le 3 variabili da unire
X=df[['gdp','Popolazione', 'urban_area']]
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)
print(f'varianza spiegata dalla prima componente {pca.explained_variance_ratio_}')
df['gdp_pop_urban_merged']=X_pca
#varianza spiegata dalla prima componente [0.87127166]

#colonne non rilevanti per la clusterizzazione
df=df.drop(columns=['hdd','cdd','prec','ele_max', 'gdp','Popolazione', 'urban_area', 'urban_area_rel'])

#esportazione
output_folder = os.path.join(cartella_corrente, 'risultati')
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'analysis_df.pkl')
df.to_pickle(output_path)