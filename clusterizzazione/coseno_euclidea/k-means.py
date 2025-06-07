import numpy as np
import os
import pandas as pd
import geopandas as gp

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..")

#importo il dataframe
pkl_path = os.path.join(cartella_progetto, "exploratory_data_analysis/normalization/risultati/analisys_df.pkl")
df = pd.read_pickle(pkl_path)
colonne_norm=['solar_pow', 'eolico', 'offshore', 'geothermal_potential']
df=df[colonne_norm]
