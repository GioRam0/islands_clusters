#si vede come le isole con alto sesonality index sono quelle con irradiazione molto bassa
import pandas as pd
import os

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))

#importo il dataframe
ris_folder = os.path.join(cartella_corrente, "..", "risultati")
pkl_path=os.path.join(ris_folder, 'analysis_df.pkl')
df = pd.read_pickle(pkl_path)
print(f'dev_stand max {df["eolico_std"].mean()}')
print(f'dev_stand min {df["eolico_std"].min()}')
print(f'dev_stand max {df["eolico_std"].max()}')
print(' ')
for i in range(1,8):
    df1=df[(df['Wind_class']==i)]
    print(f'classe {i}')
    print(f'dev_stand media {df1["eolico_std"].mean()}')
    print(f'dev_stand min {df1["eolico_std"].min()}')
    print(f'dev_stand max {df1["eolico_std"].max()}')
    print(' ')