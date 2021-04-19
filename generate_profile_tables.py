from vertical_profile import VerticalProfile
from plot_profile import plot_profile, plot_radial
import datasets.util as util
from  profile_database import ProfileDatabase
import datetime as dt
import numpy as np
import stations as stations_list
from datasets.wrf_dataset import WRFDataset
import datasets.beitdagan_sonde_dataset as beitdagan_sonde
import os

dates = [
#dt.datetime(2013, 7, 12, 18,00)
dt.datetime(2017, 9, 26, 18,00),
dt.datetime(2017,11, 25, 18,00),
dt.datetime(2018, 2, 15, 18,00),
dt.datetime(2018, 4, 30, 18,00),
#dt.datetime(2020, 9, 13, 18,00),
#dt.datetime(2020, 9, 14, 18,00),
#dt.datetime(2020, 9, 15, 18,00)
    ]

base_wrf_dir = r"E:\meteo\urban-wrf\wrfout\\"
config = 'bulk_sst'
domain = 'd03'


dataset = WRFDataset(f"{base_wrf_dir}\\{config}", domain)


get_profile(self, lat, lon, datetime, forecast_hour, minh, maxh, params)