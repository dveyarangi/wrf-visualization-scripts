import pandas as pd

import datetime as dt
from datasets.wrf_dataset import WRFDataset

import numpy as np

from  profile_database import ProfileDatabase

from mpl_toolkits.basemap import Basemap
import os

from station import WeatherStation

os.environ["PROJ_LIB"] = "D:\Dev\Anaconda3\Library\share\proj"
domains = [ "d03"]
#dt.datetime(2013, 7, 12, 18, 00), dt.datetime(2013, 7, 14, 18, 00)
time_groups = [
            #(dt.datetime(2013, 7, 12, 18, 00), dt.datetime(2013, 7, 14, 18, 00)), \
            #(dt.datetime(2013, 8, 12, 18, 00),dt.datetime(2013, 8, 14, 18, 00)), \
            #(dt.datetime(2017, 11, 25, 18, 00),dt.datetime(2017, 11, 27, 18, 00)), \
            #(dt.datetime(2018, 2, 15, 18, 00),dt.datetime(2018, 2, 17, 18, 00)), \
            #(dt.datetime(2018, 4, 30, 18, 00),dt.datetime(2018, 5, 2, 18, 00)), \
            (dt.datetime(2020, 9, 13, 18, 00), dt.datetime(2020, 9, 16, 00, 00)), \
            (dt.datetime(2020, 9, 14, 18, 00), dt.datetime(2020, 9, 17, 00, 00)), \
            (dt.datetime(2020, 9, 15, 18, 00), dt.datetime(2020, 9, 18, 00, 00)) \
 \
    ]
base_wrf_dir = r"E:\meteo\urban-wrf\wrfout\\"

configs = ['bulk_sst']

db = ProfileDatabase()
all_datasets = {}
for config in configs:
    for domain in domains:
        ds_label = f"WRF {domain} {config}"
        dataset = WRFDataset(f"{base_wrf_dir}\\{config}", domain)
        all_datasets[ds_label] = dataset
        db.register_dataset(ds_label, dataset)

dataset_labels = [f"WRF {domain} {config}"]

datasets = []

minh = 0
maxh = 3500 # height of loaded data

max_h = 2500 # height of rendered data
h_step = 100
#max_h = 700 # height of rendered data
#h_step = 25
params = ["wvel_ms", "wdir_deg", "u_ms", "v_ms", "pres_hpa", "height", "qvapor", "temp_k"]
#params = ["temp_k", "u_ms", "v_ms", "w_ms", "theta_k", "qvapor"]

for label in dataset_labels:
    datasets.append(db.get_dataset(label, minh, maxh, params))

# prepare arrays for statistics:
heights = db.get_heights(minh, maxh)

####################################################
# caching all relevant data:


tags = ['config1']

station_lat = 32.58
station_lon = 35.36
station = WeatherStation("test", station_lat, station_lon, 0)
baseOutDir = f"plots_2/config1/time_profile/{config}"
all_profiles = {}

for (start_time, end_time) in time_groups:
    ds = WRFDataset(f"{base_wrf_dir}\\{configs[0]}", domain)
    ftimes = ds.get_datetimes(start_time, end_time)
    series = ds.get_time_series(station, start_time, end_time, ['pblh'])
    data_map = {}

    times = []
    for idx, timestamp in enumerate(series.xs):
        times.append(dt.datetime.utcfromtimestamp(timestamp/1000))
    data_map['Time'] = times
    data_map['PBLH'] = series.values["pblh"]
    label_time = start_time + dt.timedelta(hours=6)
    filename = f"plots_2/csv/timeseries_{label_time.strftime('%Y%m%d')}UTC_PBLH_{station_lat}E_{station_lon}N.csv"
    print(f'Writing {filename}...')
    df = pd.DataFrame(data_map)
    df.to_csv(path_or_buf=filename)

for ds in datasets:
    for (start_time, end_time) in time_groups:

        ftimes = ds.ds.get_datetimes(start_time, end_time)

        fhours = np.zeros((len(ftimes)))
        # np.arange(0, (end_time - start_time).total_seconds() / 3600 + 1, 1)
        ref_profile = ds.get_profile(start_time, 0, station)
        ground_hgt_msl = ref_profile.heights[0]
        heights = np.arange(ground_hgt_msl, ground_hgt_msl+max_h+h_step, h_step)
        heights_labels = np.zeros((len(heights)))


        ugrid = np.zeros( (len(heights), len(ftimes)) )
        vgrid = np.zeros( (len(heights), len(ftimes)) )
        wgrid = np.zeros((len(heights), len(ftimes)))
        thgrid = np.zeros((len(heights), len(ftimes)))
        tgrid = np.zeros((len(heights), len(ftimes)))
        qgrid = np.zeros((len(heights), len(ftimes)))

        profiles = {}

        for xidx, ftime in enumerate(ftimes):


            fhours[xidx] = (ftime - start_time).total_seconds() / 3600
            p = ds.get_profile(start_time, fhours[xidx], station)


            ip = p.interpolate(heights)
            #fhours[xidx] -= 3

            profiles[fhours[xidx]] = p

        label_time = start_time + dt.timedelta(hours=6)
        ######################################################################################
 #       params = ['w_ms','u_ms','v_ms','theta_k','temp_k', 'qvapor']
        #grids = {'w_ms': wgrid, 'u_ms': ugrid, 'v_ms': vgrid, 'theta_k': thgrid, 'temp_k': tgrid}

        heights = profiles[fhours[0]].heights
        for param in params:
            data_map = {}

            profile_values = []
            for yidx, hgt in enumerate(heights):
                profile_values.append(f'{heights[yidx]:.3}')

            #data_map['Height'] = profile_values

            for xidx, ftime in enumerate(ftimes):
                ftime_str = ftime
                p = profiles[fhours[xidx]]
                profile_values = []
                for yidx, hgt in enumerate(heights):
                    profile_values.append(f'{p.values[param][yidx]:.5f}')

                data_map[ftime_str] = profile_values

            df = pd.DataFrame(data_map)

            filename = f"plots_2/csv/timecross_{label_time.strftime('%Y%m%d')}UTC_{max_h}m_{param}_{station_lat}E_{station_lon}N.csv"
            print(f'Writing {filename}...')
            df.to_csv(path_or_buf=filename)
