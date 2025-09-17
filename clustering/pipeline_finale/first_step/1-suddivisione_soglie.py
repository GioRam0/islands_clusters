#importo le librerie
import os
import pandas as pd
import numpy as np

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..", "..")

#importo il dataframe
csv_path = os.path.join(cartella_progetto, "exploratory_data_analisys/df_raw.csv")
df = pd.read_csv(csv_path)

df['dens_cluster_list']= [[] for _ in range(len(df))]
df['consumption_cluster_list']= [[] for _ in range(len(df))]
df['cluster'] = -1
soglie_den=[50,350]
soglie_consumption=[2*(10**6), 15*(10**6), 100*(10**6)]

#assegno i possibili cluster per denista
for i in range(len(soglie_den)+1):
    #limiti classi di appartenenza, non zone cuscinetto
    lower = 0 if i == 0 else soglie_den[i-1]*1.1
    upper = np.inf if i == len(soglie_den) else soglie_den[i]*0.9
    df1 = df[(df['Densità_pop'] >= lower) & (df['Densità_pop'] <= upper)]
    for ind in df1.index:
        df.loc[ind, 'dens_cluster_list'].append(i)
    if i != len(soglie_den):
        #limiti zone cuscinetto
        lower = soglie_den[i]*0.9
        upper = soglie_den[i]*1.1
        df1 = df[(df['Densità_pop'] > lower) & (df['Densità_pop'] < upper)]
        for ind in df1.index:
            #doppio cluster di possibile appartenenza
            df.loc[ind, 'dens_cluster_list'].append(i)
            df.loc[ind, 'dens_cluster_list'].append(i+1)
#assegno i possibili cluster per consumi
for i in range(len(soglie_consumption)+1):
    #limiti classi di appartenenza, non zone cuscinetto
    lower = 0 if i == 0 else soglie_consumption[i-1]*1.1
    upper = np.inf if i == len(soglie_consumption) else soglie_consumption[i]*0.9
    df1 = df[(df['consumption'] >= lower) & (df['consumption'] <= upper)]
    for ind in df1.index:
        df.loc[ind, 'consumption_cluster_list'].append(i)
    if i != len(soglie_consumption):
        #limiti zone cuscinetto
        lower = soglie_consumption[i]*0.9
        upper = soglie_consumption[i]*1.1
        df1 = df[(df['consumption'] > lower) & (df['consumption'] < upper)]
        for ind in df1.index:
            #doppio cluster di possibile appartenenza
            df.loc[ind, 'consumption_cluster_list'].append(i)
            df.loc[ind, 'consumption_cluster_list'].append(i+1)

#conto le isole rispetto alle lunghezze delle liste
condizione1 = df['dens_cluster_list'].apply(len) == 1
condizione2 = df['consumption_cluster_list'].apply(len) == 1
numero = (condizione1 & condizione2).sum()
print(f'isole non appartenenti a nessuna zona cuscinetto (assegnazione univoca): {numero}')

condizione1 = df['dens_cluster_list'].apply(len) == 1
condizione2 = df['consumption_cluster_list'].apply(len) == 2
numero = (condizione1 & condizione2).sum()
print(f'isole appartenenti alla zona cuscinetto solo per i consumi (due possibili assegnazioni): {numero}')

condizione1 = df['dens_cluster_list'].apply(len) == 2
condizione2 = df['consumption_cluster_list'].apply(len) == 1
numero = (condizione1 & condizione2).sum()
print(f'isole appartenenti alla zona cuscinetto solo per la densità (due possibili assegnazioni): {numero}')

condizione1 = df['dens_cluster_list'].apply(len) == 2
condizione2 = df['consumption_cluster_list'].apply(len) == 2
numero = (condizione1 & condizione2).sum()
print(f'isole appartenenti a entrambe le zone cuscinetto (quattro possibili assegnazioni): {numero}')

#assegno le isole con assegnazione univoca
for i, isl in df.iterrows():
    dens_list = isl.dens_cluster_list
    consumption_list = isl.consumption_cluster_list
    if len(dens_list)==1 and len(consumption_list)==1:
        value = dens_list[0] * (len(soglie_consumption) + 1) + consumption_list[0]
        df.loc[i,'cluster']=value
        continue

#esporto in pickle perche il csv tramuta le liste in stringhe 
output_folder = os.path.join(cartella_corrente, 'dataframes')
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'df_raw_assegnazioni_iniziali.pkl')
df.to_pickle(output_path)