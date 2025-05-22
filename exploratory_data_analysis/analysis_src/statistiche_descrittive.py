import pandas as pd
import os

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..")

#importo il dataframe
ris_folder = os.path.join(cartella_corrente, "..", "risultati")
pkl_path=os.path.join(ris_folder, 'analisys_df.pkl')
df = pd.read_pickle(pkl_path)
#colonne numeriche per cui non serve calcolare statistiche
colonne_da_escludere = ['ALL_Uniq', 'Wind_class', 'NO_res']
colonne_da_includere = [col for col in df.columns if col not in colonne_da_escludere]

#dataframe con statistiche descrittive delle varie colonne
descr=df[colonne_da_includere].select_dtypes(include='number').describe()

#esporto il dataframe descrittivo
output_path=os.path.join(ris_folder, 'statistiche_descrittive.xlsx')
descr.to_excel(output_path)