#importo le librerie
import numpy as np
import pickle
import os
import pandas as pd

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..")

#importo il dataframe
pkl_path = os.path.join(cartella_progetto, "exploratory_data_analisys/raw/risultati/analisys_df.pkl")
df = pd.read_pickle(pkl_path)
#importo la colonna che unifica gdp, popolazione e urbanizzazione
pkl_path = os.path.join(cartella_progetto, "exploratory_data_analysis/dimensions_reduction/risultati/analysis_df.pkl")
df1 = pd.read_pickle(pkl_path)
df['gdp_pop_urban_merged'] = df1['gdp_pop_urban_merged']

#stima dei consumi energetici
consumi={"l": 1226,
         "l-lm": 3563,
         "lm": 5901,
         "lm-um": 17341,
         "lm-um-h": 30336,
         "um-h": 42554,
         "h": 56327
         }
df['consumption']=df['Popolazione']*df['GDP_procap_etichetta'].map(consumi)

#liste di vincoli
ml_nores=[]
ml_hydro=[]
ml_geot=[]
ml_off=[]
ml_wind=[]
ml_solar=[]
cl_den=[]
cl_gdp_pc=[]
cl_dime=[]
cl_wind=[]
cl_solar=[]
cl_nores=[]

#definisco i vincoli
print(f'{len(df)} isole da svolgere')
for k,(index_isl, isola) in enumerate(df.iterrows(), 1):
    if k%50==0 or k==len(df):
        print(f'{k} isole svolte')
    for k1,(index_isl1, isola1) in enumerate(df.iterrows(), 1):
        if k1>k:
            #densita densita popolazione troppo diversa
            if isola.Densità_pop<40 and isola1.Densità_pop>=60:
                cl_den.append((min(index_isl,index_isl1), max(index_isl,index_isl1)))
                continue
            if (40<=isola.Densità_pop<60) and  isola1.Densità_pop>=350:
                cl_den.append((min(index_isl,index_isl1), max(index_isl,index_isl1)))
                continue
            if (60<=isola.Densità_pop<330) and  (isola1.Densità_pop<40 or isola1.Densità_pop>=370):
                cl_den.append((min(index_isl,index_isl1), max(index_isl,index_isl1)))
                continue
            if (330<=isola.Densità_pop<370) and  isola1.Densità_pop<50:
                cl_den.append((min(index_isl,index_isl1), max(index_isl,index_isl1)))
                continue
            if isola.Densità_pop>=370 and isola1.Densità_pop<330:
                cl_den.append((min(index_isl,index_isl1), max(index_isl,index_isl1)))
                continue

            #gdp pro capite troppo diverso
            if isola.GDP_procap_etichetta=='l' and (isola1.GDP_procap_etichetta=='h' or isola1.GDP_procap_etichetta=='um-h'):
                cl_gdp_pc.append((min(index_isl,index_isl1), max(index_isl,index_isl1)))
                continue
            if isola.GDP_procap_etichetta=='l-lm' and isola1.GDP_procap_etichetta=='h':
                cl_gdp_pc.append((min(index_isl,index_isl1), max(index_isl,index_isl1)))
                continue
            if isola.GDP_procap_etichetta=='lm' and isola1.GDP_procap_etichetta=='h':
                cl_gdp_pc.append((min(index_isl,index_isl1), max(index_isl,index_isl1)))
                continue
            if isola.GDP_procap_etichetta=='um-h' and isola1.GDP_procap_etichetta=='l':
                cl_gdp_pc.append((min(index_isl,index_isl1), max(index_isl,index_isl1)))
                continue
            if isola.GDP_procap_etichetta=='h' and (isola1.GDP_procap_etichetta=='l' or isola1.GDP_procap_etichetta=='lm' or isola1.GDP_procap_etichetta=='l-lm'):
                cl_gdp_pc.append((min(index_isl,index_isl1), max(index_isl,index_isl1)))
                continue

            #dimensioni troppo diverse
            if np.abs(isola.gdp_pop_urban_merged-isola1.gdp_pop_urban_merged)>5 and isola1.gdp_pop_urban_merged<8 and isola.gdp_pop_urban_merged<8:
                cl_dime.append((min(index_isl,index_isl1), max(index_isl,index_isl1)))
                continue

            #dimensioni abbastanza simili
            if np.abs(isola.gdp_pop_urban_merged-isola1.gdp_pop_urban_merged)<4 or (isola1.gdp_pop_urban_merged>12 and isola.gdp_pop_urban_merged>12):
                #entrambe no res
                if isola.NO_res*isola1.NO_res==1:
                    ml_nores.append((min(index_isl,index_isl1), max(index_isl,index_isl1)))
                    continue
                #hydro
                if isola.hydro>isola.consumption and isola1.hydro>isola1.consumption:
                    ml_hydro.append((min(index_isl,index_isl1), max(index_isl,index_isl1)))
                    continue
                #geothermal
                soglia = 3000
                if isola.geothermal_potential>soglia and isola1.geothermal_potential>soglia:
                    ml_geot.append((min(index_isl,index_isl1), max(index_isl,index_isl1)))
                    continue
                #offshore
                soglia = 15
                if isola.offshore>soglia and isola1.offshore>soglia:
                    ml_off.append((min(index_isl,index_isl1), max(index_isl,index_isl1)))
                    continue
                #molto sole/vento, res implementabili
                if isola.NO_res+isola1.NO_res==0:
                    if (isola.Wind_class>=5 and isola1.Wind_class>=5):
                        ml_wind.append((min(index_isl,index_isl1), max(index_isl,index_isl1)))
                        continue
                    if isola.Solar_etichetta=='L' and isola1.Solar_etichetta=='L':
                        ml_solar.append((min(index_isl,index_isl1), max(index_isl,index_isl1)))
                        continue

            #isola con prevlenza vento e l'altra senza
            if isola.Wind_class>=5 and isola.Solar_etichetta=='S' and isola1.Wind_class<4:
                cl_wind.append((min(index_isl,index_isl1), max(index_isl,index_isl1)))
                continue
            if isola1.Wind_class>=5 and isola1.Solar_etichetta=='S' and isola.Wind_class<4:
                cl_wind.append((min(index_isl,index_isl1), max(index_isl,index_isl1)))
                continue
            #isola con prevalenza sole e l'altra senza
            if isola.Solar_etichetta=='L' and isola.Wind_class<4 and isola1.Solar_etichetta=='S':
                cl_solar.append((min(index_isl,index_isl1), max(index_isl,index_isl1)))
                continue
            if isola1.Solar_etichetta=='L' and isola1.Wind_class<4 and isola.Solar_etichetta=='S':
                cl_solar.append((min(index_isl,index_isl1), max(index_isl,index_isl1)))
                continue
                        
            #una no res una si ma non unite in precedenza
            if isola.NO_res+isola1.NO_res==1:
                cl_nores.append((min(index_isl,index_isl1), max(index_isl,index_isl1)))
                continue             
ml=[ml_geot, ml_hydro, ml_off, ml_wind, ml_solar, ml_nores]
cl=[cl_den, cl_gdp_pc, cl_dime, cl_wind, cl_solar, cl_nores]
#esportazione
output_folder = os.path.join(cartella_corrente, 'preparazione')
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'must_link.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(ml, f)
output_path = os.path.join(output_folder, 'cannot_link.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(cl, f)