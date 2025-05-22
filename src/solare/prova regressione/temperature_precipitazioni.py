#importo librerie
import numpy as np
import geopandas as gp
import ee
import pickle
import os
import sys

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..")

#importo coordinate isole
isl_path=os.path.join(cartella_progetto, "data/isole_filtrate", "isole_filtrate_arro4.gpkg")
gdf = gp.read_file(isl_path)
gdf=gdf[(gdf['IslandArea']<10000)]

# percorso file config
percorso_config = os.path.join(cartella_corrente, "..", "config.py")
sys.path.append(os.path.dirname(percorso_config))
#importo la variabile project
import config
proj = config.proj
ee.Initialize(project=proj)

percorso_pkl=os.path.join(cartella_corrente, "temp.pkl")
with open(percorso_pkl, 'rb') as file:
        temp = pickle.load(file)
percorso_pkl=os.path.join(cartella_corrente, "prec.pkl")
with open(percorso_pkl, 'rb') as file:
        prec = pickle.load(file)
percorso_pkl=os.path.join(cartella_corrente, "isl_nodp.pkl")
with open(percorso_pkl, 'rb') as file:
        isl_nodp = pickle.load(file)
percorso_pkl=os.path.join(cartella_corrente, "isl_nodt.pkl")
with open(percorso_pkl, 'rb') as file:
        isl_nodt = pickle.load(file)

#scelgo il dataset e seleziono diversi anni per ridurre la varianza
dataset = ee.ImageCollection("ECMWF/ERA5/MONTHLY") \
    .filterDate("2016-06-01", "2020-05-31")
def mean_temp(image):
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean(),  # media temperature
        geometry=multip_geo,
        scale=1000,  # Risoluzione MODIS
        bestEffort=True
    )
    return image.set("mean_temp", stats.get("mean_2m_air_temperature"), "date", image.date().format())
def mean_prec(image):
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean(),  # media precipitazioni
        geometry=multip_geo,
        scale=1000,  # Risoluzione MODIS
        bestEffort=True
    )
    return image.set("mean_prec", stats.get("total_precipitation"), "date", image.date().format())

#creazione vari dizionari
#temp={}
#isl_nodt={}
#prec={}
#isl_nodp={}
k=0
for ind,isl in gdf.iterrows(): #itero per le isole
    if k % 200 == 0:
        print(k)
        percorso_file=os.path.join(cartella_corrente, "temp.pkl")
        with open(percorso_file, "wb") as f:
            pickle.dump(temp, f)
        percorso_file=os.path.join(cartella_corrente, "isl_nodt.pkl")
        with open(percorso_file, "wb") as f:
            pickle.dump(isl_nodt, f)
        percorso_file=os.path.join(cartella_corrente, "prec.pkl")
        with open(percorso_file, "wb") as f:
            pickle.dump(prec, f)
        percorso_file=os.path.join(cartella_corrente, "isl_nodp.pkl")
        with open(percorso_file, "wb") as f:
            pickle.dump(isl_nodp, f)
    k+=1
    codice=isl.ALL_Uniq
    if codice not in prec:
        multipoli=isl.geometry
        multip_list = [
            [list(vertice) for vertice in poligono.exterior.coords]
            for poligono in multipoli.geoms
        ]
        multip_geo = ee.Geometry.MultiPolygon(multip_list)
        temp_means = dataset.map(mean_temp)
        mean_list1 = temp_means.aggregate_array("mean_temp").getInfo()
        if mean_list1==[]:
            temp[codice]=np.nan
            isl_nodt[codice]=1
        else:
            #temperatura espressa in kelvin
            temp[codice]=np.mean(mean_list1)-273
            isl_nodt[codice]=0
        
        prec_means = dataset.map(mean_prec)
        mean_list2 = prec_means.aggregate_array("mean_prec").getInfo()
        if mean_list2==[]:
            prec[codice]=np.nan
            isl_nodp[codice]=1
        else:
            #faccio la media sui quattro anni
            prec[codice]=(np.sum(mean_list2))/4
            isl_nodp[codice]=0

#esportazione
percorso_file=os.path.join(cartella_corrente, "temp.pkl")
with open(percorso_file, "wb") as f:
    pickle.dump(temp, f)
percorso_file=os.path.join(cartella_corrente, "isl_nodt.pkl")
with open(percorso_file, "wb") as f:
    pickle.dump(isl_nodt, f)
percorso_file=os.path.join(cartella_corrente, "prec.pkl")
with open(percorso_file, "wb") as f:
    pickle.dump(prec, f)
percorso_file=os.path.join(cartella_corrente, "isl_nodp.pkl")
with open(percorso_file, "wb") as f:
    pickle.dump(isl_nodp, f)