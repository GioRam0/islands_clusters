import rasterio
import rasterio.mask
from shapely.geometry import box, mapping
import numpy as np
import geopandas as gp
import pickle
import os

cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..")

#importo dati consumi
percorso_folder= os.path.join(cartella_progetto, "files")
percorso_file = os.path.join(percorso_folder, "EC2019.tif")
src = rasterio.open(percorso_file)
#bordi del file
bounds = box(*src.bounds)

percorso_file=os.path.join(cartella_progetto, "data/isole_filtrate/finali", "isole.gpkg")
df = gp.read_file(percorso_file)

#conversione crs dataframe
if df.crs != src.crs:
    df = df.to_crs(src.crs)

#funzione che calcola la somma dei pixel interni a una geometria
def richiesta(multi):
    try:
        out_image, _ = rasterio.mask.mask(src, [mapping(multi)], crop=True, all_touched=True)
        no_data_value = src.nodata
        valid_pixels = out_image[(out_image != no_data_value) & (out_image != 0)]
        #se non ci sono pixel validi sollevo un errore
        if valid_pixels.size == 0:
            raise ValueError("No valid pixels found within the geometry.")
        somma = np.sum(valid_pixels)
        #indicatore binario che indica se l'isola è contenuta o meno
        indicator = int(not multi.within(bounds))
        return float(somma), indicator
    except Exception:
        return np.nan, 1

#itero per le isole applicando la funzione
consumption={} #dizionario con codici come chiavi e stima consumi come valori
no_data={} #dizionario con codici come chiavi e binario associato a no data o dati incompleti
print(f'isole da analizzare:{len(df)}')
for k,(i,isl) in enumerate(df.iterrows(),1):
    if k%250==0 or k==len(df):
        print(f'{k} isole analizzate')
    codice=isl.ALL_Uniq
    multi=isl.geometry
    sum,nodata=richiesta(multi)
    consumption[codice]=sum
    no_data[codice]=nodata

#ripeto per file gdp2019
percorso_file = os.path.join(percorso_folder, "2019GDP.tif")
src = rasterio.open(percorso_file)
bounds = box(*src.bounds)

gdp={}
no_data1={}
print(f'isole da analizzare:{len(df)}')
for k,(i,isl) in enumerate(df.iterrows(),1):
    if k%250==0 or k==len(df):
        print(f'{k} isole analizzate')
    codice=isl.ALL_Uniq
    multi=isl.geometry
    sum,nodata=richiesta(multi)
    gdp[codice]=sum
    no_data1[codice]=nodata

#esporto i risultati
output_folder = os.path.join(cartella_progetto, "data/dati_finali/gdp_consumption_2019")
os.makedirs(output_folder, exist_ok=True)
percorso_file=os.path.join(output_folder, "consumption.pkl")
with open(percorso_file, "wb") as f:
    pickle.dump(consumption, f)
percorso_file=os.path.join(output_folder, "cons_nodata.pkl")
with open(percorso_file, "wb") as f:
    pickle.dump(no_data, f)
percorso_file=os.path.join(output_folder, "gdp_2019.pkl")
with open(percorso_file, "wb") as f:
    pickle.dump(gdp, f)
percorso_file=os.path.join(output_folder, "gdp_2019_nodata.pkl")
with open(percorso_file, "wb") as f:
    pickle.dump(no_data1, f)