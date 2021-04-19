import datasets.archive_config as archive_config
import os.path
import numpy as np
import pandas as pd

import datasets.util
from spatial_index import SpatialIndex
from station import WeatherStation
from vertical_profile import VerticalProfile
import xarray as xr

class ECMWFDataset:


    def __init__(self, dataset_dir, create_filename):

        self.dataset_dir = dataset_dir
        self.create_filename = create_filename;

        if not os.path.isdir(self.dataset_dir):
            raise IOError("Cannot file ECMWF data folder %s" % archive_config.ecmwf_dir)

        # read levels to height mapping table:
        this_dir, this_filename = os.path.split(__file__)
        levels_file = os.path.join(this_dir, "ecmwf_levels.csv", )
        levels = pd.read_csv(levels_file, sep=" ", header=None)
        levels.columns = ["n", "a", "b", "ph", "pf", "gph", "hgt", "tk", "density"]
        altitudes = levels["hgt"]

        # pick a sample file
        sample_ds = None
        for subdir, dirs, files in os.walk(self.dataset_dir):
            for file in files:
                sample_ds = xr.open_dataset(os.path.join(subdir, file), engine='cfgrib')
            if sample_ds is not None:
                break
        print(sample_ds)
        z_profile = sample_ds.variables["lv_HYBL0"][:]


        self.all_hgts = np.zeros((len(z_profile)))
        for i, z in np.ndenumerate(z_profile):
            self.all_hgts[i] = altitudes[z]

        # create map of WRF points:
        lons = sample_ds.variables['lon_0'][:]
        lats = sample_ds.variables['lat_0'][:]

        self.indexer = SpatialIndex()

        for (i,lon) in np.ndenumerate( lons ):
            for (j, lat) in np.ndenumerate(lats):
                self.indexer.add( lat, lon, i, j )

    def get_profile(self, lat, lon, datetime, minh, maxh, params):
        
        # data file contains 1 day data from 0Z 
        
        if datetime.strftime("%H") == "00" : ind_h=0
        if datetime.strftime("%H") == "12" : ind_h=3
        
       
        
        path = self.create_filename( datetime )
        ds = xr.open_dataset(path, engine='cfgrib')

        (pi, pj, plat, plon) = self.indexer.get_closest_index(lat, lon)

        all_vals = {}
        times =  ds.variables["initial_time0_hours"]
        
        #print 'initial time=', int(times[0])
        
        # make sure the initial time is 0Z 
        # initial_time0_hours is hours from 1-1-1800 0Z
        
        if int(times[0]) % 24 != 0 :
            print ('WRONG initial EC time'+str( int(times[0]) % 24 )+'hours from 0Z ')
            

        for param in params:
            if param == "wvel_knt":
                ugrid = ds.variables["UGRD_P0_L105_GLL0"][:]
                vgrid = ds.variables["VGRD_P0_L105_GLL0"][:]
                
                # [0] - from 2 d array to 1d vector
                uprofile = ugrid[ind_h, :, pj, pi][0]
                vprofile = vgrid[ind_h, :, pj, pi][0]
                all_vals[param] = (uprofile ** 2 + vprofile ** 2) ** 0.5 * 1.94384  # from m/s to knot
            elif param == "wdir_deg":
                ugrid = ds.variables["UGRD_P0_L105_GLL0"][:]
                vgrid = ds.variables["VGRD_P0_L105_GLL0"][:]
                uprofile = ugrid[ind_h, :, pj, pi][0]
                vprofile = vgrid[ind_h, :, pj, pi][0]

                # TODO : verify this conversion
                # ENTER CONDITION over wind speed
                # for i in range(len(vprofile)):
                # if np.sqrt(vprofile[i]**2+uprofile[i]**2) > 2 :
                # tmp[i] = 270-np.rad2deg(np.arctan(vprofile[i]/uprofile[i]))
                # else:
                # tmp[i] = np.NaN
                all_vals[param] = datasets.util.to_degrees(uprofile, vprofile) #:270 - np.rad2deg(np.arctan(vprofile / uprofile))


            elif param == "u_knt":
                ugrid = ds.variables["UGRD_P0_L105_GLL0"][:]
                all_vals[param] = ugrid[ind_h, :, pj, pi][0]
            elif param == "v_knt":
                vgrid = ds.variables["VGRD_P0_L105_GLL0"][:]
                all_vals[param] = vgrid[ind_h, :, pj, pi][0]
            else:
                grid = ds.variables[param][:]
                all_vals[param] = grid[ind_h, :, pj, pi][0]

        size = 0
        for hgt in self.all_hgts:
            if minh <= hgt <= maxh: size = size + 1
        # convert to numpy arrays:
        hgts = np.zeros((size), dtype=float)
        vals = {}
        for param in params:
            vals[param] = np.zeros((size), dtype=float)

        idx = 0
        for all_idx, hgt in enumerate(self.all_hgts):
            if minh <= hgt <= maxh:
                hgts[idx] = hgt
                for param in params:
                    vals[param][idx] = all_vals[param][all_idx]
                idx = idx + 1
        for param in params:
            vals[param] = np.flip(vals[param],0)

        hgts = np.flip(hgts,0)
        station = WeatherStation(-1, lat, lon, 0)

        return VerticalProfile(hgts, vals, station)

    def get_station_profile(self, station, datetime, minh, maxh, param):

        return self.get_profile( station.lat, station.lon, datetime, minh, maxh, param)


def create_ECMWF_filename(self, datetime):
    return self.dataset_dir + "/" + "IS_" + datetime.strftime("%Y%m_%#d") + ".nc"

def create_ERA5_ML_filename(self, datetime):
    return f'{self.dataset_dir}/{datetime.strftime("%Y")}/era5_{datetime.strftime("%Y%m%d")}_an_ml_0.grib1'

def create_ERA5_PL_filename(self, datetime):
    return f'{self.dataset_dir}/{datetime.strftime("%Y")}/era5_{datetime.strftime("%Y%m%d")}_an_pl_0.grib1'


#dataset = ECMWFDataset()

