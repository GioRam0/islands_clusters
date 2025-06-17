#importo le librerie
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))

#importo il dataframe
pkl_path = os.path.join(cartella_corrente, "risultati","analisys_df.pkl")
df = pd.read_pickle(pkl_path)
output_filename=os.path.join(cartella_corrente, "analisys_df.csv")
df.to_csv(output_filename, index=False, encoding='utf-8')