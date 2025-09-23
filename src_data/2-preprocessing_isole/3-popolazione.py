import geopandas as gp
import os
import sys
import rasterio
import rasterio.mask
from shapely.geometry import mapping
import numpy as np

cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..")

percorso_file = os.path.join(cartella_progetto, "data/isole_filtrate/filtro_superficie/isole.gpkg")
gdf = gp.read_file(percorso_file)
print(f'lunghezza file originale: {len(gdf)}')

#importo file con dati popolazione, ha copertura globale
path_pop=os.path.join(cartella_progetto, "files", "popolazione.tif")
src = rasterio.open(path_pop)

#inizializzo le nuove colonne
gdf['Popolazione']=0
gdf['Densità_pop']=0

#itero per le isole
for k,(i,isl) in enumerate(gdf.iterrows(), 0):
    if k%1000==0 or k==len(gdf)-1:
        print(f'{k} isole svolte')
    multip=isl.geometry
    #sommo i valori dei pixel validi all'interno del multipoligono
    out_image, out_transform = rasterio.mask.mask(src, [mapping(multip)], crop=True, all_touched=True)
    no_data_value = src.nodata
    valid_pixels = out_image[out_image != no_data_value]
    pop_isola = np.sum(valid_pixels)
    #aggiorno il dataframe con i valori ottenuti
    gdf.loc[i,'Popolazione']=pop_isola
    gdf.loc[i,'Densità_pop']=pop_isola/gdf.loc[i,'IslandArea']

percorso_config = os.path.join(cartella_corrente, "..", "config.py")
sys.path.append(os.path.dirname(percorso_config))
#importo le variabili config
import config
min_pop = config.MIN_POPOLAZIONE
max_pop = config.MAX_POPOLAZIONE
#filtro
gdf_populated = gdf[(gdf['Popolazione']>max_pop)]
print(f'isole troppo popolate: {len(gdf_populated)}')
gdf=gdf[(gdf['Popolazione']>=min_pop) & (gdf['Popolazione']<=max_pop)]
print(f'isole dopo il filtro: {len(gdf)}')

#esportazione gpkg
output_folder = os.path.join(cartella_progetto, "data/isole_filtrate/filtro_popolazione")
os.makedirs(output_folder, exist_ok=True)
percorso_out = os.path.join(output_folder, "isole.gpkg")
gdf.to_file(percorso_out, driver="GPKG")

percorso_out = os.path.join(cartella_progetto, "data/isole_escluse/isole_popolate.gpkg")
gdf_populated.to_file(percorso_out, driver="GPKG")

codici=list(gdf.ALL_Uniq)
popolazioni=list(gdf.Popolazione)
densita_popolazioni=list(gdf.Densità_pop)

#ripeto il filtro ed esporto anche per il file con coordinate arrotondate
percorso_file = os.path.join(cartella_progetto, "data/isole_filtrate/filtro_superficie", "isole_arro4.gpkg")
gdf = gp.read_file(percorso_file)
print(f'lunghezza file originale: {len(gdf)}')
#elimino le isole se le ho eliminate in precedenza
for i,isl in gdf.iterrows():
    if isl.ALL_Uniq not in codici:
        gdf=gdf.drop(i)
print(f'lunghezza file dopo il filtro: {len(gdf)}')
#aggiungo la feature popolazione e densità popolazione
gdf['Popolazione'] = popolazioni
gdf['Densità_pop'] = densita_popolazioni
#esportazione gpkg
percorso_out = os.path.join(output_folder, "isole_arro4.gpkg")
gdf.to_file(percorso_out, driver="GPKG")

#ripeto il filtro ed esporto anche per il file con coordinate arrotondate a due e tre cifre decimali
percorso_file = os.path.join(cartella_progetto, "data/isole_filtrate/filtro_superficie", "isole_arro3.gpkg")
gdf = gp.read_file(percorso_file)
print(f'lunghezza file originale: {len(gdf)}')
for i,isl in gdf.iterrows():
    if isl.ALL_Uniq not in codici:
        gdf=gdf.drop(i)
print(f'lunghezza file dopo il filtro: {len(gdf)}')
gdf['Popolazione'] = popolazioni
gdf['Densità_pop'] = densita_popolazioni
percorso_out = os.path.join(output_folder, "isole_arro3.gpkg")
gdf.to_file(percorso_out, driver="GPKG")

percorso_file = os.path.join(cartella_progetto, "data/isole_filtrate/filtro_superficie", "isole_arro2.gpkg")
gdf = gp.read_file(percorso_file)
print(f'lunghezza file originale: {len(gdf)}')
for i,isl in gdf.iterrows():
    if isl.ALL_Uniq not in codici:
        gdf=gdf.drop(i)
print(f'lunghezza file dopo il filtro: {len(gdf)}')
gdf['Popolazione'] = popolazioni
gdf['Densità_pop'] = densita_popolazioni
percorso_out = os.path.join(output_folder, "isole_arro2.gpkg")
gdf.to_file(percorso_out, driver="GPKG")