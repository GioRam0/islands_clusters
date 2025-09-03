#importo le librerie
import pandas as pd
import os

#cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))

#importo il dataframe
df_folder = os.path.join(cartella_corrente, "..", "..")
csv_path=os.path.join(df_folder, 'df_raw.csv')
df = pd.read_csv(csv_path)

#colonne numeriche per cui non serve calcolare statistiche
colonne_da_escludere = ['ALL_Uniq', 'Wind_class', 'NO_res']
colonne_da_includere = [col for col in df.columns if col not in colonne_da_escludere]

#dataframe con statistiche descrittive delle varie colonne
descr=df[colonne_da_includere].select_dtypes(include='number').describe()

#esporto il dataframe descrittivo
ris_folder = os.path.join(cartella_corrente, "..", "results")
os.makedirs(ris_folder, exist_ok=True)
output_path=os.path.join(ris_folder, 'statistiche_descrittive.xlsx')
descr.to_excel(output_path)