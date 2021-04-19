import datasets.wrf_dataset as wrf
import datasets.ecmwf_dataset as ecmwf
import datasets.beitdagan_sonde_dataset as beitdagan_sonde
import datasets.wyoming_sonde_dataset as wyoming_sonde
import numpy as np
import datasets.archive_config as archive_config
import datetime as dt
 
class ProfileDatabase:




    def __init__(self):

        self.datasets = {}

        self.wrf_dataset = wrf.WRFDataset(archive_config.wrf_dir)
        self.register_dataset("WRF", self.wrf_dataset)
        #self.register_dataset("ERA5", ecmwf.ECMWFDataset())
        self.sonde_dataset = beitdagan_sonde.BeitDaganSondeDataset();
        self.register_dataset("SONDE_HIRES", self.sonde_dataset)
        #self.coarse_sonde = wyoming_sonde.WyomingSondeDataset()

    def register_dataset(self, ds_label, ds):
        self.datasets[ds_label] = ds

    def get_heights(self, minh, maxh):
        size = 0
        all_hgts = range(minh, maxh, 100)
        for hgt in all_hgts:
            if minh <= hgt <= maxh: size = size +1
        # convert to numpy arrays:
        hgts = np.zeros((size),dtype=float)

        idx = 0
        for all_idx, hgt in enumerate(all_hgts):
            if minh <= hgt <= maxh:
                hgts[idx] = hgt
                idx = idx + 1

        return hgts


    def get_profile(self, dataset_label, station, datetime, forecast_hours, minh, maxh, params ):

        ds = self.datasets[dataset_label]

        return ds.get_station_profile( station, datetime, forecast_hours, minh, maxh, params )
        try:
            #
            if "WRF" == dataset_label:
                return self.wrf_dataset.  get_station_profile( station, datetime, forecast_hours, minh, maxh, params )
            elif "ECMWF" == dataset_label:
                return self.ecmwf_dataset.get_station_profile( station, datetime, forecast_hours, minh, maxh, params)
            elif "HIRES" == dataset_label:
                return self.fine_sonde.   get_station_profile( station, datetime, forecast_hours, minh, maxh, params)
            elif "LORES" == dataset_label:
                return self.coarse_sonde. get_station_profile( station, datetime, forecast_hours, minh, maxh, params)

        except (IOError, AttributeError, ValueError) as strerror:
            print ("Failed to read %s data for %s" % (dataset_label, datetime))
            print ("%s" % strerror)
            return None

    def get_profiles(self, dataset_label, stations, datetime, minh, maxh, param):
        ds = self.datasets[dataset_label]
        ds.get_profiles( stations, datetime, minh, maxh, param)
        try:
            #
            if "WRF" == dataset_label:
                return self.wrf_dataset.get_profiles( stations, datetime, minh, maxh, param)
            elif "ECMWF" == dataset_label:
                return self.ecmwf_dataset.get_profiles( stations, datetime, minh, maxh, param)
            elif "HIRES" == dataset_label:
                return self.fine_sonde.get_profiles( stations, datetime, minh, maxh, param)
            elif "LORES" == dataset_label:
                return self.coarse_sonde.get_profiles( stations, datetime, minh, maxh, param)

        except (IOError, AttributeError, ValueError) as strerror:
            print ("Failed to read %s data for %s" % (dataset_label, datetime))
            print ("%s" % strerror)
            return None


    def get_dataset(self, dataset_label, minh, maxh, params):
        ds = self.datasets[dataset_label]
        return ProfileDataset(self, ds, dataset_label, minh, maxh, params)

    def iterator(self, datasets, height, station, min_date, max_date, forecast_hour):
        return Iterator(datasets, height, station, min_date, max_date, forecast_hour)

class ProfileDataset:

    def __init__(self, db, ds, dataset_label, minh, maxh, params):
        self.db = db
        self.ds = ds
        self.dataset_label = dataset_label

        self.minh = minh
        self.maxh = maxh
        self.params = params


    def get_profile(self, datetime, forecast_hour, station ):
        return self.db.get_profile(self.dataset_label, station, datetime, forecast_hour, self.minh, self.maxh, self.params)



class Iterator:
    def __init__(self, datasets, heights, station, min_date, max_date, forecast_hour):

        self.datasets = datasets
        self.heights = heights
        self.station = station
        self.min_date = min_date
        self.max_date = max_date
        self.curr_date = min_date
        self.forecast_hour = forecast_hour

    def __iter__(self):
        return self

    def __next__(self):

            while self.curr_date <= self.max_date:
                prev_date = self.curr_date
                self.curr_date += dt.timedelta(1)

                ps = {}
                for ds in iter(self.datasets):
                    p = ds.get_profile(prev_date, self.forecast_hour, self.station)
                    p = p.interpolate(self.heights)
                    ps[ds.dataset_label] = p


                return self.heights, ps, prev_date

            if self.curr_date > self.max_date:
                raise StopIteration


