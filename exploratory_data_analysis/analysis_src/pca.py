import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..", "..")

#importo il dataframe
ris_folder = os.path.join(cartella_corrente, "..", "risultati")
pkl_path=os.path.join(ris_folder, 'analisys_df.pkl')
df = pd.read_pickle(pkl_path)
colonne_da_escludere = ['ALL_Uniq', 'Wind_class', 'NO_res']
colonne_da_includere = [col for col in df.columns if col not in colonne_da_escludere]

#dataframe con statistiche descrittive delle varie colonne
descr=df[colonne_da_includere].select_dtypes(include='number').describe()

#analisi PCA
output_folder = os.path.join(ris_folder, 'PCA')
os.makedirs(output_folder, exist_ok=True)

col=['PC1','PC2','PC3','PC4']
X = df[colonne_da_includere].select_dtypes(include='number')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
for j in range(2,5):
    pca = PCA(n_components=j)
    X_pca = pca.fit_transform(X_scaled)
    #dataframe con componenti principali
    df_pca = pd.DataFrame(X_pca, columns=col[:j])
    print(f"Il modello spiega questa quota di varianza: {pca.explained_variance_ratio_}")
    output_path = os.path.join(output_folder, f'analisys_df_{j}_components.pkl')
    df_pca.to_pickle(output_path)
#Il modello spiega questa quota di varianza: [0.20930397 0.16931506]
#Il modello spiega questa quota di varianza: [0.20930397 0.16931506 0.09855233]
#Il modello spiega questa quota di varianza: [0.20930397 0.16931506 0.09855233 0.07016964]