import datetime as dt
from python.datasets.beitdagan_sonde_physical_dataset import BeitDaganSondeDataset

################################################
# TEST

dataset = BeitDaganSondeDataset()

datetime = dt.datetime(2016,07,01,00,00)

params = ["temp_c", "relh", "wvel_knt", "wdir_deg", "u_knt", "v_knt", "pres_hpa"]

sd = dataset.get_profile(datetime, 15000, 25000, params)


for idx,hgt in enumerate(sd.heights):
    for param in params:
        print("%dm (%s): %f"%(hgt, param, sd.values[param][idx]))


################################################
