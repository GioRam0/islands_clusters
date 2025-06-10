#importo le librerie
import numpy as np
import pickle
import os
import pandas as pd

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..")

#importo il dataframe
pkl_path = os.path.join(cartella_progetto, "exploratory_data_analisys/risultati/analisys_df.pkl")
df = pd.read_pickle(pkl_path)

ml = []
cl = []
sml = []
scl = []
sml_weights=[]
scl_weights=[]

#definisco i vincoli
for k,(index_isl, isola) in enumerate(df.iterrows(), 1):
    print(k)
    for k1,(index_isl1, isola1) in enumerate(df.iterrows(), 1):
        if k1>k:
            #densita popolazione
            if isola.Densità_pop<40 and isola1.Densità_pop>=60:
                cl.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
            if (40<=isola.Densità_pop<60) and  isola1.Densità_pop>=350:
                cl.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
            if (60<=isola.Densità_pop<330) and  (isola1.Densità_pop<40 or isola1.Densità_pop>=370):
                cl.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
            if (330<=isola.Densità_pop<370) and  isola1.Densità_pop<50:
                cl.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
            if isola.Densità_pop>=370 and isola1.Densità_pop<330:
                cl.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
            #gdp pro capite
            if isola.GDP_procap_etichetta==1 and isola1.GDP_procap_etichetta>2:
                cl.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
            if isola.GDP_procap_etichetta==2 and isola1.GDP_procap_etichetta==3:
                cl.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
            if isola.GDP_procap_etichetta==3 and isola1.GDP_procap_etichetta==1:
                cl.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
            if isola.GDP_procap_etichetta==4 and isola1.GDP_procap_etichetta<3:
                cl.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
            #sole rivedi weight
            if isola.NO_res!=1 and isola1.NO_res!=1:
                if isola.Solar_etichetta=='S' and isola.Solar_etichetta=='L':
                    scl.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
                    scl_weights.append(0.6)
                if isola.Solar_etichetta=='L' and isola.Solar_etichetta=='L':
                    sml.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
                    scl_weights.append(0.6)                    
                #vento
                if isola.Wind_class==7:
                    if isola1.Wind_class==7:
                        sml.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
                        sml_weights.append(0.5)
                    if isola1.Wind_class==6:
                        sml.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
                        sml_weights.append(0.4)
                    if isola1.Wind_class==5:
                        sml.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
                        sml_weights.append(0.3)
                    if isola1.Wind_class==2:
                        scl.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
                        scl_weights.append(0.4)
                    if isola1.Wind_class==1:
                        scl.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
                        scl_weights.append(0.5)
                if isola.Wind_class==6:
                    if isola1.Wind_class==7:
                        sml.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
                        sml_weights.append(0.4)
                    if isola1.Wind_class==6:
                        sml.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
                        sml_weights.append(0.45)
                    if isola1.Wind_class==5:
                        sml.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
                        sml_weights.append(0.4)
                    if isola1.Wind_class==4:
                        sml.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
                        sml_weights.append(0.3)
                    if isola1.Wind_class==1:
                        scl.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
                        scl_weights.append(0.4)
                if isola.Wind_class==5:
                    if isola1.Wind_class==7:
                        sml.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
                        sml_weights.append(0.3)
                    if isola1.Wind_class==6:
                        sml.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
                        sml_weights.append(0.)
                    if isola1.Wind_class==5:
                        sml.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
                        sml_weights.append(0.45)
                    if isola1.Wind_class==4:
                        sml.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
                        sml_weights.append(0.4)
                    if isola1.Wind_class==3:
                        sml.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
                        sml_weights.append(0.3)
                    if isola1.Wind_class==1:
                        scl.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
                        scl_weights.append(0.3)
                if isola.Wind_class==2:
                    if isola1.Wind_class==7:
                        scl.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
                        scl_weights.append(0.4)
                if isola.Wind_class==1:
                    if isola1.Wind_class==7:
                        scl.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
                        scl_weights.append(0.5)
                    if isola1.Wind_class==6:
                        scl.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
                        scl_weights.append(0.4)
                    if isola1.Wind_class==5:
                        scl.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
                        scl_weights.append(0.3)
            #idro rivedi soglia definisci peso
            if isola.hydro>3000 and isola1.hydro>3000:
                weight=fun(isola.hydro, isola1.hydro)
                sml.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
                sml_weights.append(weight)
            #popolazione/gdp
            #inserisci fattore normalizzazione
            if isola.Popolazione/isola1.Popolazione>2 or isola.Popolazione/isola1.Popolazione<0.5:
                scl.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
                scl_weights.append(0.5)
            if isola.Popolazione/isola1.Popolazione>3 or isola.Popolazione/isola1.Popolazione<0.33:
                scl.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
                scl_weights.append(0.7)
            if isola.GDP/isola1.GDP>2 or isola.GDP/isola1.GDP<0.5:
                scl.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
                scl_weights.append(0.5)
            if isola.GDP/isola1.GDP>3 or isola.GDP/isola1.GDP<0.33:
                scl.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
                scl_weights.append(0.7)
            #prova a cambiare valori soglia
            #prove da aggiungere per velocizzare vedi quante isole lo rispettano e come sono tra loro
            if (isola.Wind_class>=5 and isola1.Wind_class>=5) and (0.7<=isola.Popolazione/isola1.Popolazione<=(1/0.7)) and (0.7<=isola.GDP/isola1.GDP<=(1/0.7)) and (0.7<=isola.eolico_std/isola1.eolico_std<=(1/0.7)):
                ml.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])
            if (isola.Solar_etichetta=='L' and isola1.Solar_etichetta=='L') and (0.7<=isola.Popolazione/isola1.Popolazione<=(1/0.7)) and (0.7<=isola.GDP/isola1.GDP<=(1/0.7)) and (0.7<=isola.solar_seas_ind/isola1.solar_seas_ind<=(1/0.7)):
                ml.append([min(index_isl,index_isl1), max(index_isl,index_isl1)])


output_folder = os.path.join(cartella_corrente, 'vincoli')
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'must_link.pkl')
ml.to_pickle(output_path)
output_path = os.path.join(output_folder, 'cannot_link.pkl')
cl.to_pickle(output_path)
output_path = os.path.join(output_folder, 'soft_must_link.pkl')
sml.to_pickle(output_path)
output_path = os.path.join(output_folder, 'soft_cannot_link.pkl')
scl.to_pickle(output_path)
output_path = os.path.join(output_folder, 'soft_must_link_weights.pkl')
sml_weights.to_pickle(output_path)
output_path = os.path.join(output_folder, 'soft_cannot_link_weights.pkl')
scl_weights.to_pickle(output_path)