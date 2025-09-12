#importo le librerie
import numpy as np
import os
import pandas as pd

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))

#importo il dataframe
pkl_path = os.path.join(cartella_corrente, "dataframes", "df_raw_assegnazioni_iniziali.pkl")
df = pd.read_pickle(pkl_path)
df['must'] = [[] for _ in range(len(df))]

#itero per le isole, assegno un vincolo must-link per le isole non assegnate con le isole in cluster compatibili e caratteristiche analoghe
print(f'{len(df)} isole da analizzare')
for k,(i, isl) in enumerate(df.iterrows(),1):
    if k%100 == 0:
        print(f'{k} isole analizzate')
    if isl.cluster != -1:
        continue
    dens_list, cons_list = isl.dens_cluster_list, isl.consumption_cluster_list
    df1 = df[df['dens_cluster_list'].apply(lambda x: any(item in dens_list for item in x))]
    df1 = df1[df1['consumption_cluster_list'].apply(lambda x: any(item in cons_list for item in x))]
    if isl.hydro > isl.consumption:
        df1 = df1[df1['hydro'] > df1['consumption']]
        for j in df1.index:
            if j not in df.loc[i,'must']:
                df.loc[i,'must'].append(j)
            if i not in df.loc[j,'must']:
                df.loc[j, 'must'].append(i)
        continue
    if isl.NO_res == 1:
        df1 = df1[df1['NO_res'] == 1]
        for j in df1.index:
            if j not in df.loc[i,'must']:
                df.loc[i,'must'].append(j)
            if i not in df.loc[j,'must']:
                df.loc[j, 'must'].append(i)
        continue
    if isl.solar_pow >= 4.5 and isl.Wind_class > 4:
        df1 = df1[(df1['solar_pow'] >= 4.5) & (df1['Wind_class'] > 4)]
        for j in df1.index:
            if j not in df.loc[i,'must']:
                df.loc[i,'must'].append(j)
            if i not in df.loc[j,'must']:
                df.loc[j, 'must'].append(i)
        continue
    if isl.solar_pow >= 4.5 and isl.Wind_class < 3:
        df1 = df1[(df1['solar_pow'] >= 4.5) & (df1['Wind_class'] < 3)]
        for j in df1.index:
            if j not in df.loc[i,'must']:
                df.loc[i,'must'].append(j)
            if i not in df.loc[j,'must']:
                df.loc[j, 'must'].append(i)
        continue
    if isl.solar_pow <= 3.5 and isl.Wind_class > 4:
        df1 = df1[(df1['solar_pow'] <= 3.5) & (df1['Wind_class'] > 4)]
        for j in df1.index:
            if j not in df.loc[i,'must']:
                df.loc[i,'must'].append(j)
            if i not in df.loc[j,'must']:
                df.loc[j, 'must'].append(i)
        continue

#esportazione
output_folder = os.path.join(cartella_corrente, "dataframes")
output_path = os.path.join(output_folder, "df_raw_constraints.pkl")
df.to_pickle(output_path)