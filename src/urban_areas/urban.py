#nel caso si bloccasse con errore su eccessivo payload è sufficiente farlo ripartire
#importo librerie
import geopandas as gp
import ee
import pickle
import os
import sys
from shapely import MultiPolygon, Polygon

# cartella in cui si trova lo script
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.join(cartella_corrente, "..", "..")

#importo coordinate isole
isl_path=os.path.join(cartella_progetto, "data/isole_filtrate/finali", "isole_arro3.gpkg")
gdf = gp.read_file(isl_path)

# percorso file config
percorso_config = os.path.join(cartella_corrente, "..", "config.py")
sys.path.append(os.path.dirname(percorso_config))
#importo la variabile project
import config
proj = config.proj
ee.Initialize(project=proj)

#importo il dataset
urban_collection=ee.ImageCollection("JRC/GHSL/P2023A/GHS_SMOD_V2-0")
#ultima data disponibile
sorted_collection = urban_collection.sort('system:time_start', False)
last_image = sorted_collection.first()
timestamp_ms = last_image.get('system:time_start').getInfo()
import datetime
last_date = datetime.datetime.fromtimestamp(timestamp_ms/1000.0)
print(f"data dell'ultima immagine: {last_date}")
#prima data disponibile
sorted_collection = urban_collection.sort('system:time_start', True)
first_image = sorted_collection.first()
timestamp_ms = first_image.get('system:time_start').getInfo()
first_date = datetime.datetime.fromtimestamp(timestamp_ms/1000.0)
print(f"data della prima immagine: {first_date}")
#seleziono l'ultima immagine presente, non futura
filtered_collection=urban_collection.filterDate('2000-01-01', '2025-01-01')
sorted_collection=filtered_collection.sort('system:time_start', False)
urban_image=ee.Image(sorted_collection.first())
#valori dei pixel urbani
urban_values = [30, 23, 22]

#se gia presenti (effettuata una precedente run ma interrotta) importo i dati precedentemente scaricati per non ricominciare
output_folder = os.path.join(cartella_progetto, "data/dati_finali/urban")
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "urban_area.pkl")
if os.path.exists(output_path):
    with open(output_path, 'rb') as file:
        urban = pickle.load(file)
    output_path = os.path.join(output_folder, "urban_area_rel.pkl")
    with open(output_path ,  'rb') as file:
            urban_rel = pickle.load(file)
#se non presenti inizializzo i dizionari
else:
    urban={}
    urban_rel={}

gdf=gdf.sort_values(by='IslandArea', ascending=False)
print(f'isole da analizzzare: {len(gdf)}')
#itero per le isole
for k, (i, isl) in enumerate(gdf.iterrows(), 1):
    if k%100 == 0:
        print(f'{k} isole analizzate')
        print(isl.IslandArea)
    #esportazione periodica per non dover riiniziare da capo in caso di interruzione
    if k % 10 == 0:
        output_path=os.path.join(output_folder, "urban_area.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(urban, f)
        output_path=os.path.join(output_folder, "urban_area_rel.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(urban_rel, f)
    codice=isl.ALL_Uniq
    if codice not in urban:
        #semplifico le geometrie troppo grandi, o complicate
        if isl.IslandArea>30000 or isl.ALL_Uniq == 273837:
            simpli=isl.geometry.simplify(tolerance=0.005, preserve_topology=True)
            if isinstance(simpli, MultiPolygon):
                multipoli=simpli
            if isinstance(simpli, Polygon):
                multipoli=MultiPolygon([simpli])
        else:
            multipoli=isl.geometry
        multip_list =[ 
                [list(vertice) for vertice in poligono.exterior.coords]
                for poligono in multipoli.geoms
            ]   
        ee_geometry = ee.Geometry.MultiPolygon(multip_list)
        #calcolo l'area di questa figura
        area0=ee_geometry.area().getInfo()    
        #creo una maschera con i pixel urbani e la applico all'immagine ritagliata alla forma dell'isola
        clipped_image = urban_image.clip(ee_geometry)
        urban_mask = clipped_image.eq(urban_values[0])
        urban_mask = urban_mask.Or(clipped_image.eq(urban_values[1]))
        urban_mask = urban_mask.Or(clipped_image.eq(urban_values[2]))
        vectors = urban_mask.selfMask().reduceToVectors(
            geometry=urban_mask.geometry(),
            scale=clipped_image.projection().nominalScale().getInfo(),
            eightConnected=True,
            bestEffort=True
        )
        #estraggo la geometria risultante e ne calcolo la superficie
        urban_geometry = vectors.union(ee.ErrorMargin(1)).geometry()
        urban_geometry=urban_geometry.intersection(ee_geometry)
        #calcolo l'area urbana da questa figura
        urban_area = urban_geometry.area().getInfo()
        #faccio il rapporto e lo rendo percentuale
        urban_relative=(urban_area/area0)*100
        #la trasformo in km2
        urban_area=(urban_area/1000000)
        urban[codice]=urban_area
        urban_rel[codice]=urban_relative

#esportazione
output_path=os.path.join(output_folder, "urban_area.pkl")
with open(output_path, "wb") as f:
    pickle.dump(urban, f)
output_path=os.path.join(output_folder, "urban_area_rel.pkl")
with open(output_path, "wb") as f:
    pickle.dump(urban_rel, f)