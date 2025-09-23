import rasterio
import rasterio.mask
from shapely.geometry import box, mapping
import numpy as np
import geopandas as gp
import pickle
import os

cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..")

#importo dati solari
#annuali
percorso_file= os.path.join(cartella_progetto, "files", "PVOUT-year.tif")
src = rasterio.open(percorso_file)
#mensili
for i in range(1,13):
    if i<10:
        percorso_file=os.path.join(cartella_progetto, "files", f"PVOUT_month/PVOUT_0{i}.tif")
    else:
        percorso_file=os.path.join(cartella_progetto, "files", f"PVOUT_month/PVOUT_{i}.tif")
    globals()[f"src{i}"] = rasterio.open(percorso_file)
#bordi del file
bounds = box(*src.bounds)
maxx=bounds.exterior.coords[0][0]
minx=bounds.exterior.coords[2][0]
maxy=bounds.exterior.coords[1][1]
miny=bounds.exterior.coords[0][1]

percorso_file=os.path.join(cartella_progetto, "data/isole_filtrate/finali", "isole.gpkg")
gdf = gp.read_file(percorso_file)

#funzione che prende in input isola e raster e riporta il valore medio
def media(multipoligono,sr):
    out_image, _ = rasterio.mask.mask(sr, [mapping(multipoligono)], crop=True, all_touched=True)
    no_data_value = src.nodata
    valid_pixels = out_image[(out_image != no_data_value) & (out_image != 0)]
    lunghezza_np = np.count_nonzero(~np.isnan(valid_pixels))
    return np.nanmean(valid_pixels) if lunghezza_np>0 else np.nan
#funzione che prende in input l'isola e restituisce pvout medio annuale e il seasonality index come rapporto tra media mensile max e min
def richiesta(multip):
    out=media(multip, src)
    mesi=[]
    for i in range(1,13):
        mesi.append(media(multip,globals()[f"src{i}"]))
    sea_index=max(mesi)/min(mesi)
    return out,sea_index

pvout_mean={} #dizionario con codici come chiavi e insolazione media come valori
pvout_ind={} #dizionario con codici come chiavi e i seasonality indexes delle medie mensili come valori
isola_out={}
lista = [89794,273834,273876,278246,283571,284502]
print(f'isole da analizzare:{len(gdf)}')
for k,(i,isl) in enumerate(gdf.iterrows(),1):
    if k%10==0 or k==len(gdf):
        print(f'{k} isole analizzate')
    codice=isl.ALL_Uniq
    multi=isl.geometry
    #isola completamente fuori
    if multi.disjoint(bounds):
        pvout_mean[codice]=np.nan
        pvout_ind[codice]=np.nan
        isola_out[codice]=1
    else:
        out,s_ind=richiesta(multi)
        pvout_mean[codice]=out
        pvout_ind[codice]=s_ind
        if np.isnan(out):
            isola_out[codice]=1
        else:
            isola_out[codice]=0
#esportazione
percorso_folder_out = os.path.join(cartella_progetto, "data/dati_finali/solare")
os.makedirs(percorso_folder_out, exist_ok=True)
percorso_file=os.path.join(percorso_folder_out, "solar_pow.pkl")
with open(percorso_file, "wb") as f:
    pickle.dump(pvout_mean, f)
percorso_file= os.path.join(percorso_folder_out, "solar_seas_ind.pkl")
with open(percorso_file, "wb") as f:
    pickle.dump(pvout_ind, f)
percorso_file= os.path.join(percorso_folder_out, "solar_nodata.pkl")
with open(percorso_file, "wb") as f:
    pickle.dump(isola_out, f)