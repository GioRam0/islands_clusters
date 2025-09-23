import pickle
import os
import pandas as pd

cartella_corrente = os.path.dirname(os.path.abspath(__file__))

#importo i vincoli
pkl_path = os.path.join(cartella_corrente, "cannot_link_consumi.pkl")
cannot_cons = pd.read_pickle(pkl_path)
pkl_path = os.path.join(cartella_corrente, "cannot_link_densita.pkl")
cannot_densi = pd.read_pickle(pkl_path)

#conteggio vincoli singolo tipo
print(f"sono stati creati {len(cannot_cons)} cannot-link a causa di differenze di consumi")
cont1 = (len(cannot_cons)*2)/2012
print(f"un'isola mediamente ha {cont1} cannot-link a casua di differenze di consumi")
print(f"sono stati creati {len(cannot_densi)} cannot-link a causa di differenze di densità abitativa")
cont1 = (len(cannot_densi)*2)/2012
print(f"un'isola mediamente ha {cont1} cannot-link a casua di differenze di densità abitativa")

#unisco le due liste ed elimino le ripetizioni
set_unito = set(cannot_cons) | set(cannot_densi)
cannot = list(set_unito)
print(f"sono stati creati {len(cannot)} cannot-link complessivi")
cont1 = (len(cannot)*2)/2012
print(f"un'isola mediamente ha {cont1} cannot-link complessivi")
#esportazione
output_path = os.path.join(cartella_corrente, 'cannot_link.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(cannot, f)

#sono stati creati 1088608 cannot-link a causa di differenze di consumi
#un'isola mediamente ha 1082.1153081510934 cannot-link a casua di differenze di consumi
#sono stati creati 1101612 cannot-link a causa di differenze di densità abitativa
#un'isola mediamente ha 1095.0417495029822 cannot-link a casua di differenze di densità abitativa
#sono stati creati 1591096 cannot-link complessivi
#un'isola mediamente ha 1581.606361829026 cannot-link complessivi