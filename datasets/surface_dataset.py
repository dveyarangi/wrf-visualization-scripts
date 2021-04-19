import os

import numpy as np
import pandas as pd
import datetime as dt
import math
from datasets.datasets import SurfaceDataset
import datasets.util as util

from time_series import Series
import pytz

surface_archive_dir = r"D:\Dev\Machon\data\surface_archive"


DATASET_LABEL = "surface obs"

param_station = 'station'
param_datetime = 'datetime'
param_wvel_ms = 'wvel_ms'
param_wdir_deg = 'wdir_deg'
param_temp2m_k = 'temp2m_k'
param_rh = 'rh'

datetime_format = '%Y-%m-%d %H:%M:%S%z'

valid_values_max = {"wvel_ms":20, "wdir_deg":360.01, "temp2m_k":50, "rh2m":100.01}


def create_filename(archive_dir, datetime, station_name):
    return f'{archive_dir}/{datetime.strftime("%Y%m%d")}/{datetime.strftime("%Y%m%d")}_{station_name}.csv'

class SurfaceDataset(SurfaceDataset):

    def __init__(self, dataset_dir):

        self.dataset_dir = dataset_dir

        self.dataset_label = DATASET_LABEL
        if not os.path.isdir(self.dataset_dir):
            raise IOError("Cannot file WRF data folder %s" % self.dataset_dir)

        # pick a sample file
        sample_ds = None
#        for subdir, dirs, files in os.walk(self.dataset_dir):
 #           for file in files:
    def get_station_series(self, station, datetime, params):
        utc = pytz.UTC

        datetime = utc.localize(datetime)

        station_data = {}
        filename = create_filename(self.dataset_dir, datetime, station.wmoid)
        if not os.path.isfile(filename):
            print("Could not find surface data file " + filename)
            return station_data
        data = pd.read_csv(filename)
        for index, row in data.iterrows():
            datestr = row[param_datetime]
            date = dt.datetime.strptime(datestr, datetime_format)

            if date < datetime:
                continue
            if date > datetime:
                break
            temp_k = row[param_temp2m_k]
            rh = row[param_rh]
            wvel_ms = row[param_wvel_ms]
            wvel_knt = wvel_ms * 1.94384
            wdir_deg = row[param_wdir_deg]

            for param in params:

                if param == 'wvel_ms':

                    if wvel_ms < 0.5:
                        station_data[param] = None
                    else:
                        station_data[param] = wvel_ms

                elif param == 'u10_ms':
                    if wvel_ms is None or wdir_deg is None:
                        station_data[param] = None
                    else:
                        station_data[param] = -1. * wvel_ms * math.sin(math.radians(wdir_deg))

                elif param == 'v10_ms':
                    if wvel_ms is None or wdir_deg is None:
                        station_data[param] = None
                    else:
                        station_data[param] = -1. * wvel_ms * math.cos(math.radians(wdir_deg))

                elif param == 'wdir_deg':  # remove wind dir values when wind speed is < 1 m/s
                    if wvel_ms < 0.5:
                        station_data[param] = None
                    else:
                        station_data[param] = wdir_deg

                elif param == 'temp2m_c':
                    station_data[param] = temp_k - 273.15
                elif param == 'temp2m_k':
                    station_data[param] = temp_k
                elif param == 'rh':
                    station_data[param] = rh
                else:
                    station_data[param] = None


        return station_data

    def get_time_series(self, station, start_datetime, end_datetime, params):
        utc = pytz.UTC

        start_datetime = utc.localize(start_datetime)
        end_datetime = utc.localize(end_datetime)

        samples_count = 0
        total_days = (end_datetime - start_datetime).days
        idx = 0
        for day in range(0, total_days+1):

            curr_datetime = start_datetime + dt.timedelta(days=day)
            curr_datetime.replace(hour=00, minute=00, second=00)
            filename = create_filename(self.dataset_dir, curr_datetime, station.wmoid)
            if not os.path.isfile(filename):
                return None

            data = pd.read_csv(filename)
            for index, row in data.iterrows():
                datestr = row[param_datetime]
                date = dt.datetime.strptime(datestr, datetime_format)

                if date < start_datetime:
                    continue
                if date > end_datetime:
                    break

                samples_count = samples_count+1

        times = np.zeros((samples_count), dtype=np.longlong)
        vals={}
        for param in params:
            vals[param] = np.empty((samples_count), dtype=float)
            vals[param][:] = np.nan

        idx = 0
        for day in range(0, total_days+1):

            curr_datetime = start_datetime + dt.timedelta(days=day)
            curr_datetime.replace(hour=00, minute=00, second=00)
            filename = create_filename(self.dataset_dir, curr_datetime, station.wmoid)

            data = pd.read_csv(filename)
            for index, row in data.iterrows():
                datestr = row[param_datetime]
                date = dt.datetime.strptime(datestr, datetime_format)

                if date < start_datetime:
                    continue
                if date > end_datetime:
                    break

                temp_k = row[param_temp2m_k]
                if temp_k > 273.15+55: temp_k = None

                wvel_ms = row[param_wvel_ms]
                if wvel_ms > 50: wvel_ms = None

                wdir_deg = row[param_wdir_deg]
                if wdir_deg > 360.01: wdir_deg = None

                rh = row[param_rh]
                if rh > 100.01: rh = None

                times[idx] = util.unix_time_millis_localized(date)

                for param in params:

                    if param == 'wvel_ms':
                        if wvel_ms is None or wvel_ms < 0.5:
                            vals[param][idx] = None
                        else:
                            vals[param][idx] = wvel_ms

                    elif param == 'u10_ms':
                        if wvel_ms is None or wdir_deg is None:
                            vals[param][idx] = None
                        else:
                            vals[param][idx] = -1. * wvel_ms * math.sin(math.radians(wdir_deg))

                    elif param == 'v10_ms':
                        if wvel_ms is None or wdir_deg is None:
                            vals[param][idx] = None
                        else:
                            vals[param][idx] = -1. * wvel_ms * math.cos(math.radians(wdir_deg))

                    elif param == 'wdir_deg':  # remove wind dir values when wind speed is < 1 m/s
                        if wvel_ms is None or wvel_ms < 0.5:
                            vals[param][idx] = None
                        else:
                            vals[param][idx] = wdir_deg
                    elif param == 'rh2m':
                        vals[param][idx] = rh
                    elif param == 'temp2m_c':
                        if temp_k is not None:
                            vals[param][idx] = temp_k - 273.15
                        else:
                            vals[param][idx] = None
                    elif param == 'temp2m_k':
                        vals[param][idx] = temp_k
                    else:
                        vals[param][idx] = None

                idx = idx + 1


        return Series(times, vals, station) # , angular=["wdir_deg"]


def create_filename(dataset_dir, datetime, station_name):

    return f'{dataset_dir}/{datetime.strftime("%Y")}/{datetime.strftime("%m")}/{datetime.strftime("%d")}/{datetime.strftime("%Y%m%d")}_{station_name}.csv'