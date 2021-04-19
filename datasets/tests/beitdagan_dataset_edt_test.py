import datetime as dt
from python.datasets.beitdagan_sonde_edt_dataset import BeitDaganSondeEDTDataset

################################################
# TEST

dataset = BeitDaganSondeEDTDataset()


station = None
datetime = dt.datetime(2017,7,1,00,00)
forecast_offset = 0

params = ["temp_c", "relh", "wvel_knt", "wdir_deg", "u_knt", "v_knt", "pres_hpa"]

sd = dataset.get_station_profile(station, datetime, forecast_offset, 15000, 25000, params)


for idx,hgt in enumerate(sd.heights):
    for param in params:
        print("%dm (%s): %f"%(hgt, param, sd.values[param][idx]))


################################################
