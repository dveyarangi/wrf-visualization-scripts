import datetime as dt

from python.datasets.wyoming_sonde_dataset import WyomingSondeDataset

################################################
# TEST

sondeDataset = WyomingSondeDataset()

sonde_date = dt.datetime(2016,07,01,00,00)

params = ["temp_c", "relh", "wvel_knt", "wdir_deg", "u_knt", "v_knt", "pres_hpa"]

profile = sondeDataset.get_station_profile("40179", sonde_date, 15000, 25000, params)


for idx,hgt in enumerate(profile.heights):
    for param in params:
        print("%dm : %f"%(hgt, profile.values[param][idx]))
