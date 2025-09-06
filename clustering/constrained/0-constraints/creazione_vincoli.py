#importo le librerie
import pickle
import os
import pandas as pd

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..", "..")

#importo il dataframe
csv_path = os.path.join(cartella_progetto, "exploratory_data_analisys/df_raw.csv")
df = pd.read_csv(csv_path)

#liste di vincoli
cl_densi = []
cl_cons = []

for k,(ind,isl) in enumerate(df.iterrows(),0):
    if k%100==0:
        print(k)
    den = isl.Densità_pop
    cons = (isl.consumption)/1000000
    for k1,(ind1,isl1) in enumerate(df.iloc[k+1:].iterrows(),k+1):
        den1 = isl1.Densità_pop
        if den<=45:
            if den1>=55:
                cl_densi.append((ind,ind1))
        if den>45 and den<55:
            if den1>=385:
                cl_densi.append((ind,ind1))
        if den>=55 and den<=315:
            if den1 <=45 or den1 >=385:
                cl_densi.append((ind,ind1))
        if den>315 and den<385:
            if den1 <= 45:
                cl_densi.append((ind,ind1))
        if den>=385:
            if den1<=315:
                cl_densi.append((ind,ind1))

        cons1 = (isl1.consumption)/1000000
        if cons<=1.8:
            if cons1>=2.2:
                cl_cons.append((ind,ind1))
        if cons>1.8 and cons<2.2:
            if cons1>=16.5:
                cl_cons.append((ind,ind1))
        if cons>=2.2 and cons<=13.5:
            if cons1 <=1.8 or cons1 >=16.5:
                cl_cons.append((ind,ind1))
        if cons>13.5 and cons<16.5:
            if cons1 <= 1.8 or cons1 >= 110:
                cl_cons.append((ind,ind1))
        if cons>=16.5 and cons<=90:
            if cons1 <=13.5 or cons1 >=110:
                cl_cons.append((ind,ind1))
        if cons>90 and cons<110:
            if cons1 <= 13.5:
                cl_cons.append((ind,ind1))
        if cons>=110:
            if cons1<=90:
                cl_cons.append((ind,ind1))

#esportazione
output_path = os.path.join(cartella_corrente, 'cannot_link_densita.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(cl_densi, f)
output_path = os.path.join(cartella_corrente, 'cannot_link_consumi.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(cl_cons, f)