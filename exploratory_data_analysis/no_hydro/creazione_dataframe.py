#importo le librerie
import pandas as pd
import pickle
import os

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))

#importo il dataframe
folder_path = os.path.join(cartella_corrente, "../risultati")
pkl_path = os.path.join(folder_path, "analysis_df.pkl")
df = pd.read_pickle(pkl_path)

#creo una colonna con una stima dei consumi complessivi considerando sviluppo economico e popolazione
#https://ourworldindata.org/grapher/energy-use-per-person-vs-gdp-per-capita?time=2022&country=~OWID_LIC
consumi={"l": 1226,
         "l-lm": 3563,
         "lm": 5901,
         "lm-lu": 17341,
         "lm-lu-h": 30336,
         "lu-h": 42554,
         "h": 56327
         }
df['consumption']=df['Popolazione']*df['GDP_procap_etichetta'].map(consumi)
df1=df[(df['hydro']<=0.5*df['consumption'])]
df2=df[(df['hydro']>0.5*df['consumption'])]
print(f'isole con molto potenziale idroelettrico rispetto ai consumi stimati {len(df2)}')
print(f'isole con poco potenziale idroelettrico rispetto ai consumi stimati {len(df1)}')

df=df.sort_values(by='hydro', ascending=False)

#funzione per vedere dove si trovano le isole escluse rispetto all'ordinamento per idroelettrico
#utile per capire quante con molto idroelettrico sono conservate, serve inserie vincoli per queste nella clusterizzazione
lim=0
while True:
    a=list(df.iloc[0:lim]['ALL_Uniq'])
    b=list(df2['ALL_Uniq'])
    cont=0
    for el in b:
        if el in a:
            cont+=1
    print(f'nelle {lim} isole con più idorelettrico ne sono state escluse {cont}')
    print(' ')
    if cont==len(df2): break
    lim+=10

df1=df1.drop(columns=['consumption'])
df2=df2.drop(columns=['consumption'])

#esportazione
output_folder = os.path.join(cartella_corrente, 'risultati')
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'analysis_df.pkl')
df1.to_pickle(output_path)
output_path = os.path.join(output_folder, 'hydro_islands_df.pkl')
df2.to_pickle(output_path)