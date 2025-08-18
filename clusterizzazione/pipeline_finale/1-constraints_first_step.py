#importo le librerie
import numpy as np
import os
import pandas as pd
import pickle

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente,"..","..")

#importo il dataframe
pkl_folder = os.path.join(cartella_progetto, "exploratory_data_analisys/1-raw/risultati")
pkl_path = os.path.join(pkl_folder, "analisys_df.pkl")
df = pd.read_pickle(pkl_path)
df['cannot'] = [set() for _ in range(len(df))]
df['must'] = [set() for _ in range(len(df))]
df['superficie_res_assoluta'] = df['IslandArea'] * (df['superficie_res']/100)

def soglie(dataframe, val, soglie, feature):
    if val <= soglie[0]*0.9:
        return (dataframe[dataframe[feature] >= soglie[0]*1.1].index, dataframe[dataframe[feature] < soglie[0]*1.1].index)
    elif val >= soglie[-1]*1.1:
        return (dataframe[dataframe[feature] <= soglie[-1]*0.9].index, dataframe[dataframe[feature] > soglie[-1]*0.9].index)
    for i in range(len(soglie)):
        lower = (0.9*soglie[i-1]) if i > 0 else -np.inf
        upper = (1.1*soglie[i+1]) if i < len(soglie)-1 else np.inf
        if 0.9*soglie[i] < val < 1.1*soglie[i]:
            return (dataframe[(dataframe[feature] <= lower) | (dataframe[feature] >= upper)].index, dataframe[(dataframe[feature] > lower) & (dataframe[feature] < upper)].index)
        if 1.1*soglie[i] <= val <= 0.9*soglie[i+1]:
            return (dataframe[(dataframe[feature] <= 0.9*soglie[i]) | (dataframe[feature] >= 1.1*soglie[i+1])].index, dataframe[(dataframe[feature] > 0.9*soglie[i]) & (dataframe[feature] < 1.1*soglie[i+1])].index)
        
#soglie cannot link
soglie_den=[50,350]
soglie_consumption=[2*(10**6), 15*(10**6), 100*(10**6)]

#itero per le isole
print(f'{len(df)} isole da analizzare')
for k,(i, isl) in enumerate(df.iterrows(),1):
    if k%100 == 0:
        print(f'{k} isole analizzate')
    densi, consumption = isl.Densità_pop, isl.consumption
    cannot1, must = soglie(df, densi, soglie_den, 'Densità_pop')
    cannot2, must1 = soglie(df, consumption, soglie_consumption, 'consumption')
    df.loc[i, 'cannot'].update(cannot1)
    df.loc[i, 'cannot'].update(cannot2)
    must = must.intersection(must1)
    #https://www.sciencedirect.com/science/article/pii/S1364032121005803#bib25 7 m/s wind speed
    #https://www.sciencedirect.com/science/article/pii/S2352484723001397 forse estendibile soglie solari
    if isl.superficie_res_assoluta == 0:
        must = must.intersection(df[(df['superficie_res_assoluta'] == 0)].index)
        df.loc[i, 'must'].update(must)
    else:
        #https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0270155 land use
        #https://ourworldindata.org/land-use-per-energy-source
        #https://docs.google.com/spreadsheets/d/1fzYdYk_0u5ZMLHfo8OzDkCTWdlwDHLlhtzU_xTQ4esY/edit?gid=0#gid=0
        #https://unece.org/sites/default/files/2022-04/LCA_3_FINAL%20March%202022.pdf
        if isl.superficie_res >= 50:
            if isl.solar_pow >= 4.5 and isl.Wind_class > 4:
                must = must.intersection(df[(df['superficie_res'] >= 50) &
                                            (df['solar_pow'] >= 4.5) &
                                            (df['Wind_class'] > 4)].index)
                df.loc[i, 'must'].update(must)
            elif isl.Wind_class > 4:
                must = must.intersection(df[(df['superficie_res'] >= 50) &
                                            (df['solar_pow'] < 4.5) &
                                            (df['Wind_class'] > 4)].index)
                df.loc[i, 'must'].update(must)
            elif isl.solar_pow >= 4.5:
                must = must.intersection(df[(df['superficie_res'] >= 50) &
                                            (df['Wind_class'] <= 4) &
                                            (df['solar_pow'] >= 4.5)].index)
                df.loc[i, 'must'].update(must)

for i1, isl1 in df.iterrows():
    for ind in isl1.must:
        if i1 not in df.loc[ind, 'must']:
            print('prob')
#valuta meglio le soglie solar ed eolico, vedi se inserire anche hydro


#importo dataframe normalizzato
pkl_folder = os.path.join(cartella_progetto, "exploratory_data_analisys/3-normalization/risultati")
pkl_path = os.path.join(pkl_folder, "analisys_df.pkl")
df1 = pd.read_pickle(pkl_path)
df1['cannot'] = df['cannot']
df1['must'] = df['must']

#esportazione
output_folder = os.path.join(cartella_corrente, "dataframes")
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "df_raw_constraints.pkl")
df.to_pickle(output_path)
output_path = os.path.join(output_folder, "df_norm_constraints.pkl")
df1.to_pickle(output_path)