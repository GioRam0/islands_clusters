#si vede come le isole con alto sesonality index sono quelle con irradiazione molto bassa
import pandas as pd
import os

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..")

#importo il dataframe
ris_folder = os.path.join(cartella_corrente, "..", "risultati")
pkl_path=os.path.join(ris_folder, 'analisys_df.pkl')
df = pd.read_pickle(pkl_path)
print(df['solar_pow'].mean())
for i in range(4,11):
    print(i)
    df1=df[(df['solar_seas_ind']>i)]
    print(df1['solar_pow'].mean())
for i in range(10,35,5):
    print(i)
    df1=df[(df['solar_seas_ind']>i)]
    print(df1['solar_pow'].mean())
for i in range(30,200,10):
    print(i)
    df1=df[(df['solar_seas_ind']>i)]
    print(df1['solar_pow'].mean())