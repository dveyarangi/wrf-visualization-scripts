import pandas as pd
import numpy as np
from station import WeatherStation

#import
#import datasets.surface_archive as archive
import datetime as dt
import os

import datasets.surface_dataset as ds
from os import listdir
from os.path import isfile, join
import pytz


import email.utils as eutils
import time
###############################
# This program organized surface data downloaded from https://ims.data.gov.il/ims
#
#
input_dir = r'D:\Dev\Machon\data\surface_data\haifa_stat_data'

pd.set_option('display.max_columns', None)


DATETIME_KEY = 'Date & Time'
TEMP_KEY = 'TEMP'
WIND_VEL_KEY = 'WDS'
WIND_DIR_KEY = 'WDD'
RH_KEY = 'RH'
PRESSURE_KEY = 'BP'

required_keys = [TEMP_KEY, WIND_VEL_KEY, WIND_DIR_KEY]


archive_dir = r'/surface_archive'
def my_to_datetime(date_str):
    if date_str[11:13] != '24':
        return pd.to_datetime(date_str+"+02:00", format='%d/%m/%Y %H:%M%z')

    date_str = date_str[0:11] + '00' + date_str[13:]
    return pd.to_datetime(date_str+"+02:00", format='%d/%m/%Y %H:%M%z') + \
           dt.timedelta(days=1)

def to_standard_row(station, igud_file_row):
    standard_row = {}

    standard_row[ds.param_station] = station

    datetime_str = igud_file_row[DATETIME_KEY]
    datetime = my_to_datetime(datetime_str)
    #local_dt = local_tz.localize(datetime, is_dst=None)
    datetime = datetime.astimezone(pytz.utc)
    standard_row[ds.param_datetime] = datetime

    if TEMP_KEY in igud_file_row:
        temp_c = igud_file_row[TEMP_KEY]
        standard_row[ds.param_temp2m_k] = temp_c + 273.15
    else:
        standard_row[ds.param_temp2m_k] = np.nan
    if WIND_VEL_KEY in igud_file_row:
        wvel_ms = igud_file_row[WIND_VEL_KEY]
        standard_row[ds.param_wvel_ms] = wvel_ms
    else:
        standard_row[ds.param_wvel_ms] = np.nan

    if WIND_DIR_KEY in igud_file_row:
        wdir = igud_file_row[WIND_DIR_KEY]
        standard_row[ds.param_wdir_deg] = wdir
    else:
        standard_row[ds.param_wdir_deg] = np.nan
    if RH_KEY in igud_file_row:
        rh = igud_file_row[RH_KEY]
        standard_row[ds.param_rh] = rh
    else:
        standard_row[ds.param_rh] = np.nan

    return standard_row

dtypes = {
    TEMP_KEY: np.float32,
    WIND_VEL_KEY: np.float32,
    WIND_DIR_KEY: np.float32,
}

def inject_to_archive(datafile):
    print(f'Injecting {datafile}...')
    headers = pd.read_csv(datafile, nrows=1)

    metadata = headers.keys()[0].split(":")

    station_name = metadata[1].split("  ")[0].strip()

    data = pd.read_csv(datafile, skiprows=[0,1,3], skipfooter=8, dtype=dtypes, engine='python', \
                       na_values=['NoData', '<Samp', 'Zero', 'RS232', 'Down', 'InVld', 'Calm', 'FailPwr', 'Calib', 'Maintain'])

    date_standard_rows = {}

    for index, row in data.iterrows():
        standard_row = to_standard_row(station_name, row)
        date = standard_row[ds.param_datetime].date()
        if date in date_standard_rows:
            date_rows = date_standard_rows[date]
        else:
            date_standard_rows[date] = date_rows = []
        date_rows.append(standard_row)

    for date in date_standard_rows:
        rows = date_standard_rows[date]
        standard_df = pd.DataFrame(rows)

        standard_df.sort_values(ds.param_datetime)

        datetime = standard_df.iloc[0][ds.param_datetime]
        station = standard_df.iloc[0][ds.param_station]
        filename = ds.create_filename(ds.surface_archive_dir, datetime, station_name)
        dir = os.path.dirname(filename)
        if not os.path.isdir(dir):
            os.makedirs(dir)
        #print(filename)
        standard_df.to_csv(filename, float_format="%.4f")

datafiles = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
for file in datafiles:
    if file.endswith(".csv"):
        inject_to_archive(join(input_dir, file))