#importo le librerie
import pandas as pd
import pickle
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))

#importo il dataframe
folder_path = os.path.join(cartella_corrente, "../1-raw/risultati")
pkl_path = os.path.join(folder_path, "analisys_df.pkl")
df = pd.read_pickle(pkl_path)


# unisco 4 variabili con alte correlazioni reciproche
X1 = df[['gdp']].values
X2 = df[['Popolazione']].values
X3 = df['gdp_pro_capite'].values

X4 = [i*j for i in (df['gdp_pro_capite']) for j in (df['Popolazione'])]

print((X4[20]-X1[20]))
print(df.loc[20])