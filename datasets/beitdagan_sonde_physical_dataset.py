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
class BeitDaganSondePhysicalDataset(ProfileDataset):

    def __init__(self):

        self.files = []
        if not os.path.isdir(archive_config.highres_sonde_dir):
            raise IOError("Cannot file sonde data folder %s" % archive_config.highres_sonde_dir)

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
        filename = archive_config.highres_sonde_dir + "/" + \
                   date.strftime("%Y%m%d%H") + "_physical"

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
                    
                    if param == 'wvel_ms':
                        if samples[hgt]["wvel_knt"] < 0.5:
                            vals[param][idx] = None
                        else:
                            vals[param][idx] = samples[hgt]["wvel_knt"]  * 0.514444
                    
                    elif param == 'u_knt':
                        if samples[hgt]["wvel_knt"] is None or samples[hgt]["wdir_deg"] is None:
                            vals[param][idx] = None
                        else:
                            vals[param][idx] = -1.*samples[hgt]["wvel_knt"] * math.sin(math.radians(samples[hgt]["wdir_deg"]))
                            
                    elif param == 'v_knt':
                        if samples[hgt]["wvel_knt"] is None or samples[hgt]["wdir_deg"] is None:
                            vals[param][idx] = None
                        else:
                            vals[param][idx] = -1.*samples[hgt]["wvel_knt"] * math.cos(math.radians(samples[hgt]["wdir_deg"]))

                    elif param == 'wdir_deg':  # remove wind dir values when wind speed is < 1 m/s
                        if samples[hgt]["wvel_knt"]  < 0.5:
                            vals[param][idx] = None
                        else:
                            vals[param][idx] = samples[hgt][param]

                    elif param == 'temp_c':
                        vals[param][idx] = samples[hgt]["temp_c"]
                    elif param == 'temp_k':
                        vals[param][idx] = samples[hgt]["temp_c"]+273.15
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
        while not line.startswith('LOCATION'):
            line = file.readline()

        parts = line.split()
        lat_str = re.search(r'(\d+\.?\d+?).*', parts[4]).group(1)
        lon_str = re.search(r'(\d+\.?\d+?).*', parts[5]).group(1)

        station_lat = float(lat_str)
        station_lon = float(lon_str)
        # roll forward to surface obs description
        while not line.startswith('PRESSURE'):
            line = file.readline()

        surface_pressure_mb = float(line.split(":")[1].split()[0])
        station_hgt = float(file.readline().split(":")[1].split()[0])
        surface_temp_c = float(file.readline().split(":")[1].split()[0])
        surface_relh = float(file.readline().split(":")[1].split()[0])
        surface_wdir_deg = float(file.readline().split(":")[1].split()[0])
        surface_wvel_knt = float(file.readline().split(":")[1].split()[0])
        try:
            cloud_code = file.readline().split(":")[1].split()[0]
        except ValueError:
            cloud_code = '/////'
        station_number = int(file.readline().split(":")[1].split()[0])
        station_icao = file.readline().split(":")[1].split()[0]

        station = WeatherStation(station_number, station_lat, station_lon, station_hgt)

        while len(line) != 0:

            while not line.startswith('------------------------------------------------'):
                line = file.readline()

            line = file.readline()
            while not line.startswith('PHYSICAL VALUES') and len(line) != 0:
                # a single point on vertical profile,
                # represented by variable map:
                sample = {
                    "temp_c":     datasets.util.to_float(line[9:16]),
                    "relh":       datasets.util.to_float(line[16:22]),
                    "pres_hpa":   datasets.util.to_float(line[22:29]),
                    "hght_sur_m": datasets.util.to_float(line[29:37]),
                    "hght_msl_m": datasets.util.to_float(line[37:45]),
                    "wdir_deg":   datasets.util.to_float(line[62:68]),
                    "wvel_knt":   datasets.util.to_float(line[68:75]),

                }

                samples[sample["hght_msl_m"]] = sample

                line = file.readline()

        sonde_data = samples

    return (sonde_data, station)
