import datasets.beitdagan_sonde_dataset as bgsonde
from netCDF4 import Dataset
from wrf import getvar, interplevel, ll_to_xy
from vertical_profile import VerticalProfile
from plot_profile import plot_profile
import datasets.util as util
from  profile_database import ProfileDatabase
import datetime as dt
import numpy as np
from datasets.wrf_dataset import WRFDataset
import os.path

data_path = r'D:\Dev\Machon\Urban\tests\era5ml'
filename = 'wrfout_d01_2013-07-12_180000.nc'

ncfile = Dataset(f'{data_path}/{filename}')

outdir = ""

sonde_filename = r'D:\Dev\Machon\Urban\sondes\2013071300Z.txt'

(sonde, station) = bgsonde.read_sonde(sonde_filename)




wrf_tk_var = getvar(ncfile, "tk")
wrf_p_var = getvar(ncfile, "pressure")

wrf_x_y = ll_to_xy(ncfile, station.lat, station.lon)

wrf_ds = WRFDataset()

model = {}

heights = sonde.keys()
for height_msl, sonde_sample in sonde.items():
    sonde_sample = sonde[height_msl];

    pres_hpa = sonde_sample['pres_hpa'];
    sonde_temp_c = sonde_sample['temp_c'];

    wrf_temp_c_var = interplevel(wrf_tk_var, wrf_p_var, pres_hpa) - 273.15

    wrf_temp_c = wrf_temp_c_var[wrf_x_y[0], wrf_x_y[0]]
    wrf_temp_c = wrf_temp_c.values

    print(f'{pres_hpa} : {sonde_temp_c} {wrf_temp_c}')
