#importo le librerie
import rasterio
import rasterio.mask
from shapely.geometry import box, mapping
import numpy as np
import geopandas as gp
import pickle
from rtree import index
import os

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto= os.path.join(cartella_corrente, "..", "..")

#importo file con gdp_procapite delle varie zone amministrative e seleziono la colonna piu recente
file_path=os.path.join(cartella_progetto, "files", "adm2_gdp_percapita_no_continents.gpkg")
gdp_gf=gp.read_file(file_path)
gdp_gf=gdp_gf[['2022','geometry']]
#creo un elemento idx per per facilitare le iterazioni
idx = index.Index()
for k,(i, row) in enumerate(gdp_gf.iterrows(),1):
    bbox = row.geometry.bounds
    idx.insert(i, bbox)

#importo file con dati popolazione
path_pop=os.path.join(cartella_progetto, "files", "popolazione.tif")
src = rasterio.open(path_pop)

#importo coordinate isole
file_path=os.path.join(cartella_progetto, 'data/isole_filtrate/finali', 'isole_arro4.gpkg')
gdf = gp.read_file(file_path)

#dizionario da riempire con i codici come chiavi e gdp delle isole o booleano come valori
gdp={}
gdp_pro_capite={}
gdp_nodata={}

#funzione per calcolare la popolazione di una geometria
def popolazione(geometria):
    out_image, out_transform = rasterio.mask.mask(src, [mapping(geometria)], crop=True, all_touched=True)
    no_data_value = src.nodata
    valid_pixels = out_image[out_image != no_data_value]
    #sommo i valori dei pixel all'interno del multipoligono ottenendo la popolazione dell'isola e aggiungo il valore al dataframe
    pop = np.sum(valid_pixels)
    return pop

#itero per le isole
print(f'isole da svolgere: {len(gdf)}')
for k,(ind,isl) in enumerate(gdf.iterrows(),1):
    codice=isl.ALL_Uniq
    #contatore delle zone intersecate dall'isola
    h=0
    if k%250==0 or k==len(gdf):
        print(f'{k} isole analizzate')
    isola=isl.geometry
    #bounds dell'isola per trovare le zone otenzialmente intersecanti
    bbox_isola = isola.bounds
    candidati = list(idx.intersection(bbox_isola))
    for cand in candidati:
        #se la zona interseca aggiorno il contatore e salvo il valore
        zona=gdp_gf.loc[cand].geometry
        if zona.intersects(isola):
            h+=1
            gdp_pc_isola=gdp_gf.loc[cand,'2022']
    #se una zona prendo il valore salvato come gdp_procapite, lo moltiplico per la popolazione e ottengo il gdp dell'isola
    if h==1:
        gdp_pro_capite[codice]=gdp_pc_isola
        gdp[codice]=gdp_pc_isola*isl.Popolazione
        gdp_nodata[codice]=0
    #se nessuna zona non ho dati per l'isola
    if h==0:
        gdp_pro_capite[codice]=np.nan
        gdp[codice]=np.nan
        gdp_nodata[codice]=1
    #se piu di una zona calcolo il pil procapite dell'isola come media ponderata delle zone
    if h>1:
        pop_zone_isole=0
        gdp_pc_isola=0
        for cand in candidati:
            zona=gdp_gf.loc[cand].geometry
            if zona.intersects(isola):
                pop_zone_isole+=popolazione(zona)
                gdp_pc_isola+=gdp_gf.loc[cand,'2022']*popolazione(zona)
        gdp_pro_capite[codice]=gdp_pc_isola/pop_zone_isole
        gdp[codice]=(gdp_pc_isola/pop_zone_isole)*isl.Popolazione
        gdp_nodata[codice]=0

#esportazione
folder_path=os.path.join(cartella_progetto, 'data/dati_finali/gdp')
os.makedirs(folder_path, exist_ok=True)
file_path=os.path.join(folder_path, "gdp.pkl")
with open(file_path, "wb") as f:
    pickle.dump(gdp, f)
file_path=os.path.join(folder_path, "gdp_pro_capite.pkl")
with open(file_path, "wb") as f:
    pickle.dump(gdp_pro_capite, f)
file_path=os.path.join(folder_path, 'gdp_nodata.pkl')
with open(file_path, "wb") as f:
    pickle.dump(gdp_nodata, f)