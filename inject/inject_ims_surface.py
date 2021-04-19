import pandas as pd
import numpy as np
from station import WeatherStation
#import surface_archive as archive
import datetime as dt
import os
import pytz
import datasets.surface_dataset as ds
from os import listdir
from os.path import isfile, join
###############################
# This program organized surface data downloaded from https://ims.data.gov.il/ims
#
#


datafiles = [
             #r'D:\Dev\Machon\data\surface_data\ims_data_haifa_20130712-20130719.csv',
             #r'D:\Dev\Machon\data\surface_data\ims_data_haifa_20161012-20161019.csv',
             #r'D:\Dev\Machon\data\surface_data\ims_data_haifa_20170926-20171002.csv',
             #r'D:\Dev\Machon\data\surface_data\ims_data_haifa_20171125-20171202.csv',
             #r'D:\Dev\Machon\data\surface_data\ims_data_haifa_20180214-20180505.csv',
             #r'D:\Dev\Machon\data\surface_data\ims_data_haifa_20200913-20200920.csv',
                r'D:\Dev\Machon\data\surface_data\ims_data_haifa_20200913-20200920.csv',
            ]
input_dir = r'D:\Dev\Machon\data\surface_data\IMS_haifa_csv\\'

DATE_KEY = 'תאריך'
TIME_KEY = 'שעה- LST'


TEMP_KEY = 'טמפרטורה(C°)'
WIND_VEL_KEY = 'מהירות הרוח(m/s)'
WIND_DIR_KEY = 'כיוון הרוח(מעלות)'
RH_KEY = 'לחות יחסית(%)'
local_tz = pytz.timezone('Etc/GMT+3')


STATION_NAME_KEY = 'שם תחנה'
DATE_KEY = 'תאריך'
stations = {
        'חיפה טכניון': 'Haifa Technion',
        'חיפה אוניברסיטה': 'Haifa University',
        'חיפה בתי זיקוק': 'Haifa Refineries',
        'אפק': 'Afek',
        'עין כרמל': 'Ein Karmel',
        'שבי ציון': 'Shavei Zion',
        'בית דגן': 'Beit Dagan',
        'עין החורש': 'Ein Horesh',
        'כפר בלום': 'Kfar Blum',
        'אשדוד נמל':'Ashdod Port',
        'אשקלון נמל':'Ashkelon Port',
        'בשור חווה':'Besor',
        'בית דגן':'Beit Dagan',
        'דורות':'Dorot'
}

archive_dir = r'/surface_archive'


def to_standard_row(ims_file_row):
    standard_row = {}

    ims_station_name = ims_file_row[STATION_NAME_KEY].strip()
    station = stations[ims_station_name]
    standard_row[ds.param_station] = station

    date_str = ims_file_row[DATE_KEY]
    time_str = ims_file_row[TIME_KEY]
    datetime = dt.datetime.strptime(date_str + ' ' + time_str+"+02:00", "%d-%m-%Y %H:%M%z")
    #local_dt = local_tz.localize(datetime, is_dst=None)
    datetime = datetime.astimezone(pytz.utc)

    standard_row[ds.param_datetime] = datetime

    try:
        temp_c = ims_file_row[TEMP_KEY]
        standard_row[ds.param_temp2m_k] = temp_c + 273.15
        wvel_ms = ims_file_row[WIND_VEL_KEY]
        standard_row[ds.param_wvel_ms] = wvel_ms
        wdir = ims_file_row[WIND_DIR_KEY]
        standard_row[ds.param_wdir_deg] = wdir
        rh = ims_file_row[RH_KEY]
        standard_row[ds.param_rh] = rh
    except KeyError:
        return standard_row

    return standard_row

dtypes = {
    TEMP_KEY: np.float32,
    WIND_VEL_KEY: np.float32,
    WIND_DIR_KEY: np.float32,
}

def inject_to_archive(datafile):

    data = pd.read_csv(datafile, encoding="ISO-8859-8", dtype=dtypes, na_values=['-'])

    standard_stations_data = {}
    print(f"Processing {datafile}...")
    station_dfs = [x for _, x in data.groupby(data[STATION_NAME_KEY])]
    for station_df in station_dfs:
        station_name = station_df.sample(n=1).iloc[0][STATION_NAME_KEY]
        station_name = stations[station_name.strip()]

        if station_name in standard_stations_data:
            date_standard_rows = standard_stations_data[station_name]
        else:
            date_standard_rows = {}
            standard_stations_data[station_name] = date_standard_rows


        #date_dfs = [x for _, x in station_df.groupby(data[DATE_KEY])]
        for index, row in station_df.iterrows():
            standard_row = to_standard_row(row)
            date = standard_row[ds.param_datetime].date()
            if date in date_standard_rows:
                date_rows = date_standard_rows[date]
            else:
                date_standard_rows[date] = date_rows = []
            date_rows.append(standard_row)

    for station_name in standard_stations_data:

        station_data = standard_stations_data[station_name]
        for date in station_data:
            rows = station_data[date]
            standard_df = pd.DataFrame(rows)

            standard_df.sort_values(ds.param_datetime)

            datetime = standard_df.iloc[0][ds.param_datetime]
            station = standard_df.iloc[0][ds.param_station]
            filename = ds.create_filename(ds.surface_archive_dir, datetime, station_name)
            dir = os.path.dirname(filename)
            if not os.path.isdir(dir):
                os.makedirs(dir)
            print(filename)
            standard_df.to_csv(filename, encoding="ISO-8859-8")


datafiles = [os.path.join(dp, f) for dp, dn, filenames in os.walk(input_dir) for f in filenames if f.endswith('.csv')]

for file in datafiles:
    if file.endswith(".csv"):
        inject_to_archive(join(input_dir, file))
#pd.set_option('display.max_columns', None)

#for file in datafiles:
#    inject_to_archive(file)