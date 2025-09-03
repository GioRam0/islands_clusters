#importo le librerie
import pandas as pd
import pickle
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))

#importo il dataframe
folder_path = os.path.join(cartella_corrente, "..")
csv_path = os.path.join(folder_path, "df_raw.csv")
df = pd.read_csv(csv_path)

# unisco 4 variabili con alte correlazioni reciproche
X=df[['gdp_2019','Popolazione', 'urban_area', 'consumption']]
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)
print(f'varianza spiegata dalla prima componente {pca.explained_variance_ratio_}')
df['gdp_cons_pop_urban_merged']=X_pca
#varianza spiegata dalla prima componente [0.79485721]

#considero superficie e superficie res in unico parametro
df['superficie_res']=(df['superficie_res']/100)*df['IslandArea']
#colonne non rilevanti per la clusterizzazione
df=df.drop(columns=['hdd','cdd','prec','ele_max','gdp_2019','consumption','Popolazione', 'urban_area', 'urban_area_rel','IslandArea'])

#esportazione
output_folder = os.path.join(cartella_corrente, '..')
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'df_dim_reduction.csv')
df.to_csv(output_path, index=False, encoding='utf-8')