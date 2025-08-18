#importo le librerie
import pandas as pd
import pickle
import os

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))

#importo il dataframe
pkl_path = os.path.join(cartella_corrente, "risultati","analisys_df.pkl")
df = pd.read_pickle(pkl_path)
output_filename=os.path.join(cartella_corrente, "risultati", "analisys_df.csv")
df.to_csv(output_filename, index=False, encoding='utf-8')