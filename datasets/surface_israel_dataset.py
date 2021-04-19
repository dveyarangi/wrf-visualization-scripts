import os

import numpy as np
import pandas as pd
import datetime as dt
import math
from datasets.datasets import SurfaceDataset
from station import WeatherStation
from time_series import Series
import datasets.util as util
import pytz

DATE_KEY = 'תאריך'
TIME_KEY = 'שעה- LST'


TEMP_KEY = 'טמפרטורה(C°)'
WIND_VEL_KEY = 'מהירות הרוח(m/s)'
WIND_DIR_KEY = 'כיוון הרוח(מעלות)'
RH_KEY = 'לחות יחסית(%)'
local_tz = pytz.timezone("Asia/Jerusalem")

DATASET_LABEL = "surface obs"

dtypes = {
    TEMP_KEY: np.float32,
    WIND_VEL_KEY: np.float32,
    WIND_DIR_KEY: np.float32
}


class IsraelSurfaceDataset(SurfaceDataset):


    def __init__(self, dataset_dir):

        self.dataset_dir = dataset_dir

        self.dataset_label = DATASET_LABEL
        if not os.path.isdir(self.dataset_dir):
            raise IOError("Cannot file WRF data folder %s" % self.dataset_dir)

        # pick a sample file
        sample_ds = None
#        for subdir, dirs, files in os.walk(self.dataset_dir):
 #           for file in files:

    def get_time_series(self, station, start_datetime, end_datetime, params):

        start_datetime = pytz.UTC.localize(start_datetime)
        end_datetime = pytz.UTC.localize(end_datetime)

        datetimes = []

        total_days = (end_datetime - start_datetime).days
        for day in range(0, total_days+2):

            curr_datetime = start_datetime + dt.timedelta(days=day)
            curr_datetime.replace(hour=00, minute=00, second=00)
            filename = create_filename(self.dataset_dir, curr_datetime, station.wmoid)

            data = pd.read_csv(filename, encoding="ISO-8859-8", dtype=dtypes, na_values=['-'])
            time_dfs = [x for _, x in data.groupby(data[TIME_KEY])]
            for time_df in time_dfs:
                date_str = time_df.iloc[0][DATE_KEY]
                time_str = time_df.iloc[0][TIME_KEY]
                date = dt.datetime.strptime(date_str + ' ' + time_str, "%d-%m-%Y %H:%M")
                local_dt = local_tz.localize(date, is_dst=None)
                date = local_dt.astimezone(pytz.utc)
                if date < start_datetime:
                    continue
                if date > end_datetime:
                    break

                datetimes.append(date)

        vals = {}
        times = np.zeros((len(datetimes)), dtype=np.longlong)
        vals={}
        for param in params:
            vals[param] = np.zeros((len(datetimes)), dtype=float)


        idx = 0
        for day in range(0, total_days + 2):
            curr_datetime = start_datetime + dt.timedelta(days=day)
            curr_datetime.replace(hour=00, minute=00, second=00)
            filename = create_filename(self.dataset_dir, curr_datetime, station.wmoid)

            data = pd.read_csv(filename, encoding="ISO-8859-8", dtype=dtypes, na_values=['-'])
            time_dfs = [x for _, x in data.groupby(data[TIME_KEY])]
            for time_df in time_dfs:
                date_str = time_df.iloc[0][DATE_KEY]
                time_str = time_df.iloc[0][TIME_KEY]
                curr_time = dt.datetime.strptime(date_str + ' ' + time_str, "%d-%m-%Y %H:%M")
                local_dt = local_tz.localize(curr_time, is_dst=None)
                curr_time = local_dt.astimezone(pytz.utc)

                if curr_time < start_datetime:
                    continue
                if curr_time > end_datetime:
                    break
                temp_c = time_df.iloc[0][TEMP_KEY]
                wvel_ms = time_df.iloc[0][WIND_VEL_KEY]
                wvel_knt = wvel_ms * 1.94384
                wdir = time_df.iloc[0][WIND_DIR_KEY]

                times[idx] = util.unix_time_millis_localized(curr_time)

                for param in params:

                    if param == 'wvel_ms':
                        if wvel_ms < 0.5:
                            vals[param][idx] = None
                        else:
                            vals[param][idx] = wvel_ms

                    elif param == 'u10_ms':
                        if wvel_ms is None or wdir is None:
                            vals[param][idx] = None
                        else:
                            vals[param][idx] = -1. * wvel_ms * math.sin(math.radians(wdir))

                    elif param == 'v10_ms':
                        if wvel_ms is None or wdir is None:
                            vals[param][idx] = None
                        else:
                            vals[param][idx] = -1. * wvel_ms * math.cos(math.radians(wdir))

                    elif param == 'wdir_deg':  # remove wind dir values when wind speed is < 1 m/s
                        if wvel_ms < 0.5:
                            vals[param][idx] = None
                        else:
                            vals[param][idx] = wdir

                    elif param == 'temp2m_c':
                        vals[param][idx] = temp_c
                    elif param == 'temp2m_k':
                        vals[param][idx] = temp_c + 273.15
                    else:
                        vals[param][idx] = None

                idx = idx + 1


        return Series(times, vals, station)


def create_filename(dataset_dir, datetime, station_name):

    return f'{dataset_dir}/{datetime.strftime("%Y%m%d")}/{datetime.strftime("%Y%m%d")}_{station_name}.csv'