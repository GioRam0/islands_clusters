#importo le librerie
import os
import pickle
import geopandas as gp
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..")

#importo i dati di consumo e gdp precedentemente calcolati
folder_pkl=os.path.join(cartella_progetto, "data/dati_finali/gdp_consumption_2019")
percorso_pkl = os.path.join(folder_pkl, "consumption.pkl")
with open(percorso_pkl, 'rb') as file:
    consumption = pickle.load(file)
percorso_pkl=os.path.join(folder_pkl, "cons_nodata.pkl")
with open(percorso_pkl, 'rb') as file:
    cons_no_data = pickle.load(file)
percorso_pkl=os.path.join(folder_pkl, "gdp_2019.pkl")
with open(percorso_pkl, 'rb') as file:
    gdp_2019 = pickle.load(file)
percorso_pkl=os.path.join(folder_pkl, "gdp_2019_nodata.pkl")
with open(percorso_pkl, 'rb') as file:
    gdp2019_no_data = pickle.load(file)

folder_pkl=os.path.join(cartella_progetto, "data/dati_finali/gdp")
percorso_pkl = os.path.join(folder_pkl, "gdp.pkl")
with open(percorso_pkl, 'rb') as file:
    gdp = pickle.load(file)
percorso_pkl=os.path.join(folder_pkl, "gdp_nodata.pkl")
with open(percorso_pkl, 'rb') as file:
    gdp_no_data = pickle.load(file)
# importo il dataframe
percorso_folder=os.path.join(cartella_progetto, "data/isole_filtrate/finali")
percorso_file = os.path.join(percorso_folder, "isole_arro2.gpkg")
df = gp.read_file(percorso_file)
print(len(df))
df['consumption']=np.zeros(len(df))
df['gdp_2019']=np.zeros(len(df))
df['gdp']=np.zeros(len(df))
cont=[0,0,0,0]
prob=[0,0,0,0,0,0]
for k,(ind,isl) in enumerate(df.iterrows(),1):
    codice=isl.ALL_Uniq
    if cons_no_data[codice]==0 and gdp2019_no_data[codice]==0:
        cont[0]+=1
    if cons_no_data[codice]==1 and gdp2019_no_data[codice]==0:
        cont[1]+=1
    if cons_no_data[codice]==0 and gdp2019_no_data[codice]==1:
        cont[2]+=1
    if cons_no_data[codice]==1 and gdp2019_no_data[codice]==1:
        cont[3]+=1
    
    if cons_no_data[codice]==0 and np.isnan(consumption[codice]):
        prob[0]+=1
    if cons_no_data[codice]==1 and (not np.isnan(consumption[codice])):
        prob[1]+=1
    if gdp2019_no_data[codice]==0 and np.isnan(gdp_2019[codice]):
        prob[2]+=1
    if gdp2019_no_data[codice]==1 and (not np.isnan(gdp_2019[codice])):
        prob[3]+=1
    if gdp_no_data[codice]==0 and np.isnan(gdp[codice]):
        prob[4]+=1
    if gdp_no_data[codice]==1 and (not np.isnan(gdp[codice])):
        prob[5]+=1

    if gdp2019_no_data[codice]+cons_no_data[codice]+gdp2019_no_data[codice]==0:
        df.loc[ind,'consumption']=consumption[codice]
        df.loc[ind,'gdp_2019']=gdp_2019[codice]
        df.loc[ind,'gdp']=gdp[codice]
    else:
        df = df.drop(ind)
print(cont)
print(prob)
print(len(df))

df = df.dropna(subset=['consumption', 'gdp_2019', 'gdp'], how='any')
print(len(df))

X = df[['gdp']].values
y = df['gdp_2019'].values

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

print(f"R2 between 'gdp' and 'gdp_2019': {r2:.4f}")

X = df[['gdp_2019']].values
y = df['gdp'].values

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

print(f"R2 between 'gdp_2019' and 'gdp': {r2:.4f}")

X = df[['gdp']].values
y = df['consumption'].values

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

print(f"R2 between 'gdp' and 'consumption': {r2:.4f}")

X = df[['consumption']].values
y = df['gdp_2019'].values

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

print(f"R2 between 'consumption' and 'gdp_2019': {r2:.4f}")

X = df[['gdp','gdp_2019']].values
y = df['consumption'].values

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

print(f"R2 between 'gdp','gdp_2019' and 'consumption': {r2:.4f}")