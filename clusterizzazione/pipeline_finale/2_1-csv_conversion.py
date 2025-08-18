#importo le librerie
import os
import pandas as pd
import pickle
# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..")

#importo il dataframe
pkl_path = os.path.join(cartella_corrente, 'dataframes/df_norm_first_step.pkl')
df = pd.read_pickle(pkl_path)

csv_path = os.path.join(cartella_corrente, 'dataframes/df_norm_first_step.csv')
df.to_csv(csv_path, index=False)