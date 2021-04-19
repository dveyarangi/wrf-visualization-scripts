import numpy as np
from collections import OrderedDict

import python.datasets.util
from station import WeatherStation
from vertical_profile import VerticalProfile

import python.datasets.archive_config as archive_config

################################################
# this class provides access to low resolution sonde dataset
#
class WyomingSondeDataset2:

    ################################################
    # get profile for specified station wmoid
    def get_station_profile(self, wmoid, datetime, minh, maxh, param):

        # load sonde data:
        (samples, station) = self.load_sonde(wmoid, datetime)

        # convert to numpy arrays:
        # calculate size:
        size = 0
        for hgt in samples.iterkeys():
            if minh <= hgt <= maxh: size = size + 1

        # create arrays:
        hgts = np.zeros((size), dtype=float)
        vals = np.zeros((size), dtype=float)

        # fill arrays:
        idx = 0
        for hgt in samples.iterkeys():
            if minh <= hgt <= maxh:
                hgts[idx] = hgt
                vals[idx] = samples[hgt][param]
                idx = idx + 1
        # providing resulting profile:
        return VerticalProfile(hgts, vals, station)


    ################################################
    # initialize dataset
    def __init__(self):

       self.datasetDir = archive_config.wyoming_sonde_dir

    ################################################
    # load sonde for specified date and station id
    def load_sonde(self, wmoid, datetime):
        filepath = self.datasetDir + "/" + datetime.strftime("%Y-%m") + "/" + \
                   datetime.strftime("%d") + "/" + \
            datetime.strftime("%Y-%m-%d_%HZ") + "_" + str(wmoid) + "_sounding.txt"

        sonde_data = self.read_sonde( filepath )

        return sonde_data

    ################################################


    def read_sonde(self, sonde_filename):

        samples = OrderedDict()
        station = None

        with open( sonde_filename, 'r') as file:
            # skip header:
            file.readline(); file.readline(); file.readline(); file.readline()

            # read profile
            line = file.readline()
            while not line.startswith('</PRE>') :

                # a single point on vertical profile,
                # represented by variable map:
                sample = {
                    "pres_hpa"   : python.datasets.util.to_float(line[0:8]),
                    "hght_msl_m" : python.datasets.util.to_float(line[9:15]),
                    "temp_c"     : python.datasets.util.to_float(line[16:22]),
                    "dwpt_c"     : python.datasets.util.to_float(line[23:29]),
                    "relh"       : python.datasets.util.to_float(line[30:36]),
                    "mixr_gkg"   : python.datasets.util.to_float(line[37:43]),
                    "wdir_deg"   : python.datasets.util.to_float(line[44:50]),
                    "wvel_knt"   : python.datasets.util.to_float(line[51:58]),
                    "thta_k"     : python.datasets.util.to_float(line[58:64]),
                    "thte_k"     : python.datasets.util.to_float(line[65:71]),
                    "thtv_k"     : python.datasets.util.to_float(line[72:78])
                }

                samples[sample["hght_msl_m"]] = sample

                line = file.readline()

            # extract sonde station information:
            line = file.readline()
            line = file.readline()
            if line.startswith("Station identifier:"):
                station_icao = line.split(":")[1]
                line = file.readline()
            station_number = int(line.split(":")[1])
            obs_time = file.readline().split(":")[1]
            station_lat = float(file.readline().split(":")[1])
            station_lon = float(file.readline().split(":")[1])
            station_hgt = float(file.readline().split(":")[1])

            station = WeatherStation(station_number, station_lat, station_lon, station_hgt)

        return samples, station

