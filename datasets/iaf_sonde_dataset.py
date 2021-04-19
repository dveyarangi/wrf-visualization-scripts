import pandas as pd
import numpy as np
import math
from collections import OrderedDict

import datetime as dt
from datetime import timedelta

import archive_config
import python.datasets.util
from station import WeatherStation
from vertical_profile import VerticalProfile


class IafSondeDataset:

    def __init__(self):

        self.df = pd.read_csv(archive_config.iaf_sonde_dir + '\\2016.txt', header=0)

        # drop unneeded fields:
        self.df = self.df[['BLOCK_STATION','OBS_TIME','PRESSURE','HEIGHT','TEMPERATURE','DEWPOINT_TEMP','WIND_DIRECTION','WIND_SPEED']]

    ################################################
    # get profile for specified station wmoid
    def get_station_profile(self, wmoid, datetime, minh, maxh, params):

        # load sonde data:
        samples = self.read_sonde(wmoid, datetime)

        station = station = WeatherStation(wmoid, 0, 0, 0)

        # convert to numpy arrays:
        # calculate size:
        size = 0
        for hgt in samples.iterkeys():
            if minh <= hgt <= maxh: size = size + 1

        # create arrays:
        hgts = np.zeros((size), dtype=float)
        vals = {}
        for param in params:
            vals[param] = np.zeros((size), dtype=float)
            vals[param][:] = np.nan

        # fill arrays:
        idx = 0
        for hgt in samples.iterkeys():
            if minh <= hgt <= maxh:
                hgts[idx] = hgt
                for param in params:
                    if param == 'u_knt':

                        if (samples[hgt]["wdir_deg"] is None or samples[hgt]["wvel_knt"] is None):
                            vals[param][idx] = None
                        else:
                            vals[param][idx] = samples[hgt]["wvel_knt"] * math.cos(
                                math.radians(270 - samples[hgt]["wdir_deg"]))



                    elif param == 'v_knt':
                        if (samples[hgt]["wdir_deg"] is None or samples[hgt]["wvel_knt"] is None):
                            vals[param][idx] = None
                        else:
                            vals[param][idx] = samples[hgt]["wvel_knt"] * math.sin(
                                math.radians(270 - samples[hgt]["wdir_deg"]))

                    elif param == 'wdir_deg':  # remove wind dir values when wind speed is < 1 m/s
                        if (samples[hgt]["wvel_knt"] < 2):
                            vals[param][idx] = None
                        else:
                            vals[param][idx] = samples[hgt][param]


                    else:
                        vals[param][idx] = samples[hgt][param]

                idx = idx + 1
        # providing resulting profile:
        return VerticalProfile(hgts, vals, station)

    def read_sonde(self, wmoid, datetime):
        epoch_seconds = (datetime - dt.datetime(1970, 1, 1)).total_seconds()

        sonde_df = self.df.loc[self.df['BLOCK_STATION'] == long(epoch_seconds)]

        sonde_df = sonde_df.loc[sonde_df['OBS_TIME'] == int(wmoid)]
        hour_offset = 1
        while sonde_df.empty and hour_offset <=5:
            datetime = datetime + timedelta(hours=1)
            hour_offset += 1
            epoch_seconds = (datetime - dt.datetime(1970, 1, 1)).total_seconds()
            print epoch_seconds
            sonde_df = self.df.loc[self.df['OBS_TIME'] == long(epoch_seconds)]

        if sonde_df.empty:
            return None

        samples = OrderedDict()

        for index, row in sonde_df.iterrows():

            pres = python.datasets.util.to_float(row['PRESSURE'])
            if np.isnan(pres):
                continue
            hgt = python.datasets.util.to_float(row['HEIGHT'])
            if np.isnan(hgt):
                hgt = (10.0**(math.log(pres/1000) / 5.2558797) - 1.0) / (-6.8755856 * 10**-6)


            sample = {
                "pres_hpa": pres,
                "hght_msl_m": hgt,
                "temp_c": python.datasets.util.to_float('TEMPERATURE'),
                "dwpt_c": python.datasets.util.to_float('DEWPOINT_TEMP'),
                "wdir_deg": python.datasets.util.to_float('WIND_DIRECTION'),
                "wvel_knt": python.datasets.util.to_float('WIND_SPEED'),

            }
            samples[hgt] = sample

        return samples


