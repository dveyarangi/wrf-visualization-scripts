import os.path

import datasets.util
from station import WeatherStation
from collections import OrderedDict
from vertical_profile import VerticalProfile
import datasets.archive_config as archive_config
import math
import re
import datetime as dt
import numpy as np
from datasets.datasets import ProfileDataset


################################################
# this class provides access to Beit Dagan
# high resolution sonde dataset
#
class BeitDaganSondeEDTDataset(ProfileDataset):

    def __init__(self):

        self.files = []
        self.edt_path = archive_config.highres_sonde_dir + "/EDT/"
        if not os.path.isdir(self.edt_path):
            raise IOError("Cannot file sonde data folder %s" % self.edt_path)

    def get_profiles(self, stations, datetime, forecast_hour, minh, maxh, param):

        profiles = {}
        for station in stations:
            if station.wmoid == 40179:
                station_profile = self.get_profile(station.wmoid, datetime, forecast_hour, minh, maxh, param)
                profiles[station] = station_profile
        return profiles

    ################################################
    # Retrieve a sonde from specified date
    #
    def get_station_profile(self, station, date, forecast_hours, minh, maxh, params):

        date = date + dt.timedelta(hours=forecast_hours)
        # create filename:
        filename = self.edt_path + "/" + \
                   date.strftime("%Y") + "/" + date.strftime("%Y%m%d%H") + "_EDT"

        (samples, station) = read_sonde( filename )

        size = 0
        for hgt in iter(samples):
            if minh <= hgt <= maxh: size = size +1
        # convert to numpy arrays:
        hgts = np.zeros((size),dtype=float)
        vals = {}
        
        for param in params:
            vals[param] = np.zeros((size), dtype=float)
            vals[param][:] = np.nan
            
            
        idx = 0
        for hgt in iter(samples):
            if minh <= hgt <= maxh:
                hgts[idx] = hgt
                for param in params:
                    
                    if param == 'wvel_knt':
                        if samples[hgt]["wvel_ms"] < 1:
                            vals[param][idx] = None
                        else:
                            vals[param][idx] = samples[hgt]["wvel_ms"] * 1.94384
                    
                    elif param == 'u_knt':
                        if samples[hgt]["u_ms"] is None:
                            vals[param][idx] = None
                        else:
                            vals[param][idx] = samples[hgt]["u_ms"] * 1.94384
                            
                    elif param == 'v_knt':
                        if samples[hgt]["v_ms"] is None:
                            vals[param][idx] = None
                        else:
                            vals[param][idx] = samples[hgt]["v_ms"] * 1.94384

                    elif param == 'wdir_deg':  # remove wind dir values when wind speed is < 1 m/s
                        if samples[hgt]["wvel_ms"]  < 1:
                            vals[param][idx] = None
                        else:
                            vals[param][idx] = samples[hgt][param]
                    elif param == 'temp_k':
                        vals[param][idx] = samples[hgt]["temp_k"]
                    elif param == 'temp_c':
                        vals[param][idx] = samples[hgt]["temp_k"]+273.15
                    else:
                        vals[param][idx] = samples[hgt][param]

                idx = idx+1

        return VerticalProfile(hgts, vals, station)


################################################
# read sonde from a file into VerticalDataset
#
def read_sonde(sonde_filename):

    sonde_data = None
    samples = OrderedDict()
    station = None

    with open( sonde_filename, 'r') as file:
        # skip header:
        line = file.readline()
        while not line.startswith('Station:'):
            line = file.readline()


        parts = line.split()
        station_number = parts[1]
        station_lat = None
        station_lon = None

        # roll forward to data table
        while not line.startswith('**************'):
            line = file.readline()
        line = file.readline() # skip empty line
        line = file.readline() # skip titles line
        line = file.readline() # skip titles line

        while len(line) != 0:

            parts = line.split()

            if len(parts) == 0:
                break

            if station_lat is None:
                station_lon = datasets.util.to_float(parts[14])
                station_lat = datasets.util.to_float(parts[15])

            # a single point on vertical profile,
            # represented by variable map:
            u_ms = datasets.util.to_float(parts[5])
            v_ms = datasets.util.to_float(parts[4])
            sample = {
                "time": datasets.util.to_float(parts[0]),
                "temp_k": datasets.util.to_float(parts[2]),
                "relh": datasets.util.to_float(parts[3]),
                "pres_hpa": datasets.util.to_float(parts[7]),
                #                "hght_sur_m": datasets.util.to_float(line[29:37]),
                "hght_msl_m": datasets.util.to_float(parts[6]),
                "u_ms": u_ms,
                "v_ms": v_ms,
                "wdir_deg": (math.atan2(u_ms, v_ms) * (180. / math.pi) + 180) % 360, #(270. - (math.atan2(u_ms, v_ms) * (180. / math.pi))) % 360.,
                "wvel_ms": (u_ms ** 2 + v_ms ** 2) ** 0.5,
            }

            samples[sample["hght_msl_m"]] = sample
            line = file.readline()

        sonde_data = samples

        station = WeatherStation(station_number, station_number, None, None, None)

    return (sonde_data, station)
