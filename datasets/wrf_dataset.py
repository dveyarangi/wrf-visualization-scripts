import numpy as np
import os.path
import math

import datasets.util as util
import datasets
from datasets.datasets import ProfileDataset
from datasets.datasets import SurfaceDataset
from spatial_index import SpatialIndex
from station import WeatherStation
from vertical_profile import VerticalProfile
import datetime as dt
import re
from time_series import Series
import wrf

class WRFDataset(ProfileDataset, SurfaceDataset):

    def __init__(self, dataset_dir, domain="d01"):

        self.dataset_dir = dataset_dir
        self.domain = domain


        if not os.path.isdir(self.dataset_dir):
            raise IOError("Cannot file WRF data folder %s" % self.dataset_dir)

        # pick a sample file
        sample_ds = None
        for subdir, dirs, files in os.walk(self.dataset_dir):
            for file in files:
                if file.startswith(f"wrfout_{domain}_"):
                    sample_ds = util.load_dataset(os.path.join(subdir, file))
                if sample_ds is not None:
                    break

        elevation_grid = (sample_ds.variables["PH"][:] + sample_ds.variables["PHB"][:]) / 9.81
        self.all_hgts = elevation_grid[0,:,0,0]

        # create map of WRF points:
        self.lons = sample_ds.variables['XLONG'][:]
        self.lats = sample_ds.variables['XLAT'][:]

        self.indexer = SpatialIndex()

        for ((n,i,j),val) in np.ndenumerate( self.lons ):
            self.indexer.add( self.lats[n,i,j], self.lons[n,i,j], i, j )

    def get_station_profile(self, station, datetime, forecast_offset, minh, maxh, param ):

        return self.get_profile( station.lat, station.lon, datetime, forecast_offset, minh, maxh, param)

    def get_profiles(self, stations, datetime, forecast_offset, minh, maxh, param ):

        profiles = {}
        for station in stations:
            station_profile = self.get_profile( station.lat, station.lon, datetime, forecast_offset, minh, maxh, param)
            profiles[station] = station_profile
        return profiles


    def get_domain_coverage(self):
        lonmin = self.lons[0,0,0]
        lonmax = self.lons[0,0,-1]
        latmin = self.lats[0,0,0]
        latmax = self.lats[0,-1,0]
        return (latmin,lonmin,latmax,lonmax)

    def get_profile(self, lat, lon, datetime, forecast_hour, minh, maxh, params):

        path = self.create_filename(datetime, forecast_hour*60)

        ds = datasets.util.load_dataset(path)

        times = ds.variables["Times"][:];
        (pi,pj,plat,plon) = self.indexer.get_closest_index( lat, lon )

        elevation_grid = (ds.variables["PH"][:] + ds.variables["PHB"][:])/9.81

        all_hgts = elevation_grid[0,:,pi,pj]

        all_vals = {}

        for param in params:
            if param == "wvel_ms":
                ugrid = ds.variables["U"][:]
                vgrid = ds.variables["V"][:]
                uprofile = ugrid[0, :, pi, pj]
                vprofile = vgrid[0, :, pi, pj]
                all_vals[param] = (uprofile**2+vprofile**2)**0.5  # from m/s to knot
            elif param == "wdir_deg":
                ugrid = ds.variables["U"][:]
                vgrid = ds.variables["V"][:]
                uprofile = ugrid[0, :, pi, pj]
                vprofile = vgrid[0, :, pi, pj]
                 
                
                # TODO : verify this conversion
                # ENTER CONDITION over wind speed
                #for i in range(len(vprofile)):
                    #if np.sqrt(vprofile[i]**2+uprofile[i]**2) > 2 :                    
                        #tmp[i] = 270-np.rad2deg(np.arctan(vprofile[i]/uprofile[i]))
                    #else:
                        #tmp[i] = np.NaN
                
                all_vals[param]=  (270. - ( np.arctan2(vprofile,uprofile)*(180./math.pi) ))%360. 
                
                
            elif param == "u_ms":
                ugrid = ds.variables["U"][:]
                all_vals[param] = ugrid[0, :, pi, pj]
            elif param == "v_ms":
                vgrid = ds.variables["V"][:]
                all_vals[param] = vgrid[0, :, pi, pj]
            elif param == "u_knt":
                ugrid = ds.variables["U"][:]
                all_vals[param] = ugrid[0, :, pi, pj] * 1.94384
            elif param == "v_knt":
                vgrid = ds.variables["V"][:]
                all_vals[param] = vgrid[0, :, pi, pj] * 1.94384
            elif param == "w_ms":
                wgrid = ds.variables["W"][:]
                all_vals[param] = wgrid[0, :, pi, pj]
            elif param == "height":
                all_vals[param] = all_hgts
            elif param == "qvapor":
                qgrid = ds.variables["QVAPOR"][:]
                all_vals[param] = qgrid[0, :, pi, pj]
            elif param == "pres_hpa":
                pgrid = ds.variables["PB"][:] + ds.variables["P"][:]
                all_vals[param] = pgrid[0, :, pi, pj]/100.0
            elif param == "temp_c" or param == "temp_k":
                tgrid = ds.variables["T"][:]
                pgrid = ds.variables["P"][:]
                pbgrid = ds.variables["PB"][:]

                all_vals[param] = (tgrid[0, :, pi, pj] + 300.0) * (
                            1. / 100000 * (pgrid[0, :, pi, pj] + pbgrid[0, :, pi, pj])) ** (2. / 7)
                if param == "temp_c":
                    all_vals[param] = all_vals[param] - 273.15

            elif param == "theta_k":
                tgrid = ds.variables["T"][:]
                all_vals[param] = (tgrid[0, :, pi, pj] + 300.0)
            elif param == "rh2m":
                rhgrid = wrf.rh(ds.variables["Q2"][:], ds.variables["PSFC"][:], ds.variables["T2"][:])
                all_vals[param] = rhgrid[0, :, pi, pj]
            elif param == "rh":
                tgrid = ds.variables["T"][:]
                pgrid = ds.variables["P"][:]
                pbgrid = ds.variables["PB"][:]
                qgrid = ds.variables["QVAPOR"][:]
                tk = (tgrid[0, :, pi, pj] + 300.0) * (
                        1. / 100000 * (pgrid[0, :, pi, pj] + pbgrid[0, :, pi, pj])) ** (2. / 7)
                pTot = (ds.variables["PB"][0, :, pi, pj]+ds.variables["P"][0, :, pi, pj])
                all_vals[param] = wrf.rh(qgrid[0, :, pi, pj], pTot, tk)
            else:
                grid = ds.variables[param][:]
                all_vals[param] = grid[0,:,pi,pj]

        size = 0
        for hgt in all_hgts:
            if minh <= hgt <= maxh: size = size +1
        # convert to numpy arrays:
        hgts = np.zeros((size),dtype=float)
        vals = {}
        for param in params:
            vals[param] = np.zeros((size), dtype=float)

        idx = 0
        for all_idx, hgt in enumerate(all_hgts):
            if minh <= hgt <= maxh:
                hgts[idx] = hgt
                for param in params:
                    vals[param][idx] = all_vals[param][all_idx]
                idx = idx + 1

        station = WeatherStation(-1, -1, lat, lon, 0)

        return VerticalProfile(hgts, vals, station)

    def create_folder(self, basetime):

        return f'{self.dataset_dir}/{basetime.strftime("%Y%m%d%H")}'


    def create_filename(self, datetime, forecast_minute):

        ftime = datetime + dt.timedelta(minutes=forecast_minute)
        return f'{self.dataset_dir}/{datetime.strftime("%Y%m%d%H")}/wrfout_{self.domain}_' \
               f'{ftime.strftime("%Y-%m-%d_%H^%%%M")}^%00'


    def get_file_info(self, filename):

        groups = re.search(r'wrfout_(d\d\d)_(\d\d\d\d-\d\d-\d\d_\d\d\^%\d\d\^%\d\d)', filename)
        if groups is None:
            return None, None
        domain = groups.group(1)
        datetime_str = groups.group(2).replace("^%", ":")
        datetime = dt.datetime.strptime(datetime_str, "%Y-%m-%d_%H:%M:%S")

        return domain, datetime

    def get_datetimes(self, start_time, end_time):
        datetimes = []
        base_time_dir = self.create_folder(start_time);

        for subdir, dirs, files in os.walk(base_time_dir):
            for file in files:
                (domain, datetime) = self.get_file_info(file)
                if domain is None:
                    print(f"Ignoring file {base_time_dir}/{subdir}/{file}")
                    continue

                if not domain == self.domain:
                    continue

                if datetime < start_time or datetime > end_time:
                    continue

                datetimes.append(datetime)

        if len(datetimes) == 0:
            raise IOError(f'Missing data in {base_time_dir}')

        return datetimes

    def get_time_series(self, station, start_time, end_time, params):

        (pi,pj,plat,plon) = self.indexer.get_closest_index( station.lat, station.lon )

        datetimes = self.get_datetimes(start_time, end_time)

        times = np.zeros((len(datetimes)), dtype=np.longlong)
        vals={}
        for param in params:
            vals[param] = np.empty((len(datetimes)), dtype=float)
            vals[param][:] = np.nan

        idx = 0
        for curr_time in datetimes:

            forecast_minute = (curr_time - start_time).total_seconds()/60
            path = self.create_filename(start_time, forecast_minute)
            #print(f'{curr_time} - {path}')
            ds = util.load_dataset(path)
            if ds is None:
                continue

            times[idx] = util.unix_time_millis(curr_time)

            for param in params:

                if param == "u10_ms":
                    ugrid = ds.variables["U10"][:]
                    vals[param][idx] = ugrid[0, pi, pj]
                elif param == "v10_ms":
                    vgrid = ds.variables["V10"][:]
                    vals[param][idx] = vgrid[0, pi, pj]
                elif param == "u10_knt":
                    ugrid = ds.variables["U10"][:]
                    vals[param][idx] = ugrid[0, pi, pj] * 1.94384
                elif param == "v10_knt":
                    vgrid = ds.variables["V10"][:]
                    vals[param][idx] = vgrid[0, pi, pj] * 1.94384
                elif param == "temp2m_k":
                    vgrid = ds.variables["T2"][:]
                    vals[param][idx] = vgrid[0, pi, pj]
                elif param == "wvel_ms":
                    ugrid = ds.variables["U10"][:]
                    vgrid = ds.variables["V10"][:]
                    uprofile = ugrid[0, pi, pj]
                    vprofile = vgrid[0, pi, pj]
                    vals[param][idx] = (uprofile**2+vprofile**2)**0.5
                elif param == "wdir_deg":
                    ugrid = ds.variables["U10"][:]
                    vgrid = ds.variables["V10"][:]
                    uprofile = ugrid[0, pi, pj]
                    vprofile = vgrid[0, pi, pj]

                    vals[param][idx] = (270. - (np.arctan2(vprofile, uprofile) * (180. / math.pi))) % 360.
                elif param == "pblh":
                    pblh = ds.variables["PBLH"][:]
                    vals[param][idx] = pblh[0, pi, pj]
                elif param == "rh2m":
                    rhgrid = wrf.rh(ds.variables["Q2"][:], ds.variables["PSFC"][:], ds.variables["T2"][:])
                    vals[param][idx] = rhgrid[0, pi, pj]

            idx = idx + 1

        return Series(times, vals, station, angular=[]) #angular=["wdir_deg"]


