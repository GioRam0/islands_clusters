import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

cartella_corrente = os.path.dirname(os.path.abspath(__file__))

df_folder = os.path.join(cartella_corrente, "..", "..")
csv_path=os.path.join(df_folder, 'df_raw.csv')
df = pd.read_csv(csv_path)
#colonne numeriche non da analizzare
colonne_da_escludere = ['ALL_Uniq', 'Wind_class', 'NO_res']
colonne_da_includere = [col for col in df.columns if col not in colonne_da_escludere]

#cartella esportazione
ris_folder = os.path.join(cartella_corrente, "..", "results")
os.makedirs(ris_folder, exist_ok=True)
output_folder = os.path.join(ris_folder, 'PCA')
os.makedirs(output_folder, exist_ok=True)

#analisi PCA
col=['PC1','PC2','PC3','PC4']
X = df[colonne_da_includere].select_dtypes(include='number')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#ripeto per un diverso numero di componenti principali
for j in range(2,5):
    pca = PCA(n_components=j)
    X_pca = pca.fit_transform(X_scaled)
    #dataframe con componenti principali
    df_pca = pd.DataFrame(X_pca, columns=col[:j])
    print(f"Il modello spiega questa quota di varianza: {pca.explained_variance_ratio_}")
    output_path = os.path.join(output_folder, f'analisys_df_{j}_components.pkl')
    df_pca.to_pickle(output_path)

#Il modello spiega questa quota di varianza: [0.24179813 0.19489437]
#Il modello spiega questa quota di varianza: [0.24179813 0.19489437 0.09125106]
#Il modello spiega questa quota di varianza: [0.24179813 0.19489437 0.09125106 0.06864923]