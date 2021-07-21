#############################################################################
#
# This script requires WRF out files to be placed using specific pattern:
#   {dataset_dir}\{initialization_time_dir}\{wrffile}
# for example:
#   D:\data\2016101218\wrfout_d01_2016-10-13_00^%00^%00
#
#############################################################################


import datetime as dt
from datasets.wrf_dataset import WRFDataset
from vertical_profile import VerticalProfile
from plot_profile import plot_profile

###########################
# Dataset folder
base_wrf_dir = r'E:\meteo\urban-wrf\wrfout\\'
config = 'bulk_sst'
dataset_dir = f'{base_wrf_dir}\\{config}'

###########################
# prepare dataset utility
domain = 'd03'
dataset = WRFDataset(dataset_dir, domain)

###########################
# profile settings
lat = 32.7736
lon = 35.0223
initialization_time = dt.datetime(2017, 9, 26, 18,00)
forecast_hours = 6
(minh, maxh) = 0, 5000
params = ["wvel_ms", "wdir_deg", "temp_k"]

###########################
# extract profiles
# you may want to look at and redact this method to use additional variables
profiles = dataset.get_profile( lat, lon, initialization_time, forecast_hours, minh, maxh, params)

param = "temp_k"
heights = profiles.heights
profile = profiles.values[param]

print(heights)
print(profile)

###########################
# plot profiles:
xlim=270,300
title= f"{param} at {initialization_time} (+{forecast_hours}h"

outdir = "D:" # r"D:\out" #put a value here to generate image file (i.e
prefix = f'vertical_profile_{config}_{initialization_time.strftime("%Y%m%d%H")}+{forecast_hours}_{domain}_{param}'
labeled_profiles = {config:profile} # labeled profiles to plot
plot_profile(VerticalProfile(heights, labeled_profiles, None, False),None, outdir, xlim=xlim, title=title, prefix=prefix )

