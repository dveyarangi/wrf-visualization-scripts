import pandas as pd

import datetime as dt
from datasets.wrf_dataset import WRFDataset
import datasets.surface_dataset as sid
from datasets.surface_israel_dataset import IsraelSurfaceDataset
import datasets.util as util
import numpy as np
from time_series import Series
from plot_profile import plot_time_series
import stations as st
import os
#start_time = dt.datetime(2013, 7, 13, 00, 00)
#end_time = dt.datetime(2013, 7, 13, 12, 00)

station_names = [ \
                 'Afek', \
                 'Ein Karmel', \
                 #'Haifa Refineries', \
                 'Haifa Technion', \
                 'Haifa University', \
                 'Shavei Zion', \
'Nesher','K.HAIM-REGAVIM','K.Hasidim','K.Bnyamin','K.Tivon','K.Yam','Shprinzak','N.Shanan','Ahuza','K.Bialik-Ofarim','IGUD','K.Ata'
                ]

station_names.extend([
    "Ein Karmel North",
    "Haifa Technion West",
    "Haifa University South",
    "Haifa University West",
    "Shavei Zion East",
    "Nesher South",
    "K.Hasidim East",
    "N.Shanan South",
    "N.Shanan West",
    "Ahuza East",
    "K.Bialik-Ofarim South",
    "K.Bialik-Ofarim East",
    "K.Ata East",
])


all_stations = st.load_surface_stations_map('etc/stations_and_neighbors.csv')
stations = [all_stations[name] for name in station_names]

#domain_groups = [["d01"], ["d02"], ["d03", "d04"]]
domain_groups = [["d04"]]
#dt.datetime(2013, 7, 12, 18, 00), dt.datetime(2013, 7, 14, 18, 00)
time_groups = [
            (dt.datetime(2013, 7, 12, 18, 00), dt.datetime(2013, 7, 14, 18, 00)), \
            #(dt.datetime(2013, 8, 12, 18, 00),dt.datetime(2013, 8, 14, 18, 00)), \
            (dt.datetime(2017, 9, 26, 18, 00),dt.datetime(2017, 9, 28, 18, 00)), \
            (dt.datetime(2017, 11, 25, 18, 00),dt.datetime(2017, 11, 27, 18, 00)), \
            (dt.datetime(2018, 2, 15, 18, 00),dt.datetime(2018, 2, 17, 18, 00)), \
            (dt.datetime(2018, 4, 30, 18, 00),dt.datetime(2018, 5, 2, 18, 00)), \
            (dt.datetime(2020, 9, 13, 18, 00),dt.datetime(2020, 9, 15, 18, 00)), \
            (dt.datetime(2020, 9, 14, 18, 00),dt.datetime(2020, 9, 16, 18, 00)), \
            (dt.datetime(2020, 9, 15, 18, 00),dt.datetime(2020, 9, 17, 18, 00)), \
            (dt.datetime(2016, 10, 12, 18, 00), dt.datetime(2016, 10, 15, 00, 00)),
            #(dt.datetime(2018, 4, 29, 18, 00), dt.datetime(2018, 5, 2, 18, 00))

]

tags = ['config1']

#print(obs_series)

params = ["wvel_ms", "wdir_deg", "temp2m_k", "u10_ms", "v10_ms", "rh2m"]

param_ranges={"wvel_ms":(0, 15), "wdir_deg":(0, 360), "temp2m_k":(280, 310), "u_ms":(0, 50), "v_ms":(0,50), "rh2m":(0,100)}


surface_ds = sid.SurfaceDataset(sid.surface_archive_dir)
configs = ['bulk_sst'] #
tag_datasets = {}

for tag in tags:
    domain_datasets = {sid.DATASET_LABEL: surface_ds}
    base_wrf_dir = r"E:\meteo\urban-wrf\wrfout\\"
    for domain in ["d01", "d02", "d03", "d04"]:
        for cfg in configs:
            domain_datasets [f"{cfg} {domain.upper()}"] = WRFDataset(f"{base_wrf_dir}\\{cfg}", domain)
    tag_datasets[tag] = domain_datasets


def create_plots(start_date, end_date, tag, config, domain_group, station):

    outdir = f'plots/{tag}/series/'
    os.makedirs(outdir, exist_ok=True)
    #station = sid.stations[wmoId]
    #stations = [station]

    all_datasets = tag_datasets[tag]

    domain_label = "-".join(domain_group)

    # wrfpld04_series = wrfpld04.get_time_series(station,  start_time, end_time,  params)
    dataset_labels = [sid.DATASET_LABEL]
    for domain in domain_group:
        for cfg in configs:
            dataset_labels.append(f"{cfg} {domain.upper()}")

    datasets = []

    for ds_label in dataset_labels:
        datasets.append(all_datasets[ds_label])

    ref_ds = surface_ds


    ####################################################
    # caching all relevant data:

    #print("Caching profiles...")

    ref_series = all_datasets[sid.DATASET_LABEL].get_time_series(station, start_date, end_date, params)
    if ref_series is None:
        print(f"Missing data for {station.wmoid} {start_date}")
        return
    timesNum = len(ref_series.xs)



    curr_series = {}
    for ds_label in dataset_labels:
        dataset = all_datasets[ds_label]
        series = dataset.get_time_series(station, start_date, end_date, params)
        series = series.interpolate(ref_series.xs)
        curr_series[ds_label] = series
    all_series = [(curr_series, start_date)]


    if len(all_series) == 0:
        print("No data found")
        exit()


    all_values = {}
    all_bias = {}
    all_mae = {}
    all_rmse = {}

    all_mean = {}

    all_var = {}
    all_var2 = {}
    all_count = {}  # at each z level number of events  when model and measurement exist
    all_delta = {}
    for ds_label in dataset_labels:

        values = all_values[ds_label] = {}
        bias = all_bias[ds_label] = {}
        mae = all_mae[ds_label] = {}
        rmse = all_rmse[ds_label] = {}
        mean = all_mean[ds_label] = {}

        var = all_var[ds_label] = {}
        var2 = all_var2[ds_label] = {}
        count = all_count[ds_label] = {}
        delta = all_delta[ds_label] = {}

        for param in params:
            values[param] = np.zeros((timesNum))
            values[param][:] = np.nan
            bias[param] = np.zeros((timesNum))
            mae[param] = np.zeros((timesNum))
            rmse[param] = np.zeros((timesNum))
            mean[param] = np.zeros((timesNum))
            mean[param] = np.zeros((timesNum))

            var[param] = np.zeros((timesNum))
            var2[param] = np.zeros((timesNum))

            count[param] = np.zeros((timesNum))
            delta[param] = np.zeros((timesNum))
            delta[param][:] = np.nan

    #for (heights, profiles, curr_date) in all_profiles.values():
    #    for ds_label in iter(profiles):
    #        profile = profiles[ds_label]
     #       count = all_count[ds_label]
    #        for param in params:
    #            for ix in range(len(heights)):
    #                # data was found and this is not a missing value
    #                # for wind dir nan for missing data or when sp < 2 knt. wind dir values - are wrong
    #                # have to set correct values according to u v
    #                if (profile.values[param] is not None and not np.isnan(profile.values[param][ix])): count[param][ix] += 1

    ####################################################
    #  params = ["wvel_knt", "wdir_deg", "u_knt", "v_knt", "pres_hpa"]
    #   1 kt = 0.51444444444 mps

    #print("Calculating statistics...")
    for ( profiles, curr_date) in all_series:
        surface = ref_ds.get_time_series(station, start_date, end_date, params)
        for ds_label in iter(profiles):
            model = profiles[ds_label]

            values = all_values[ds_label]
            bias = all_bias[ds_label]
            mae = all_mae[ds_label]
            rmse = all_rmse[ds_label]
            mean = all_mean[ds_label]

            var = all_var[ds_label]
            var2 = all_var2[ds_label]
            count = all_count[ds_label]
            delta = all_delta[ds_label]

            for param in params:
                model_values = model.values[param]  # type: np array
                surface_values = surface.values[param]  # type: np array
                # print param
                # print model_values
                # print sonde_values
                if model_values is None or surface_values is None:
                    continue
                delta = np.zeros((timesNum))
                delta[:] = np.nan
                if param == "wdir_deg":
                    for ix in range(timesNum):
                        if model_values[ix] is not np.nan and surface_values[ix] is not np.nan:
                            delta[ix] = util.to_degrees(surface.values["u10_ms"][ix], surface.values["v10_ms"][ix]) \
                                        - \
                                        util.to_degrees(model.values["u10_ms"][ix], model.values["v10_ms"][ix])

                            delta[ix] = util.wrap_degrees(delta[ix])

                else:
                    if len(model_values) == 0 or len(model_values) == 0 :
                        print(model_values)
                    for ix in range(timesNum):
                        if model_values[ix] is not np.nan and surface_values[ix] is not np.nan:
                            delta[ix] = surface_values[ix] - model_values[ix]

                for ix in range(timesNum):
                    if not np.isnan(delta[ix]):
                        count[param][ix] += 1
                    else:
                        delta[ix] = 0  # delta = 0 , do not increase number of events
                # print count[param]

                bias[param] += delta

                for ix in range(timesNum):
                    values[param][ix] = model_values[ix]
                    if not np.isnan(surface_values[ix]):
                        mean[param][ix] += model_values[ix]

                mae[param] += abs(delta)
                rmse[param] += delta ** 2

            for param in params:
                if param != "wdir_deg":
                    for ix in range(timesNum):
                        if count[param][ix] != 0:
                            mean[param][ix] /= count[param][ix]

            # print sonde_mean[param]
            for ix in range(timesNum):
                if count[param][ix] != 0:
                    bias[param][ix] /= count[param][ix]
                    mae[param][ix] /= count[param][ix]
                    rmse[param][ix] = (rmse[param][ix] / count[param][ix]) ** 0.5

            for ix in range(timesNum):
                if (not np.isnan(mean["u10_ms"][ix])):
                    mean["wdir_deg"][ix] = util.to_degrees(mean["u10_ms"][ix], mean["v10_ms"][ix])

    # completed mean bias ame rmse calculations
    # print sonde_mean["wdir_deg"]


    ####################################################
    # second pass: calculate variance


    for (profiles, curr_date) in all_series:
        sonde = ref_ds.get_time_series(station, start_date, end_date, params)
        for ds_label in iter(profiles):
            model = profiles[ds_label]

            bias = all_bias[ds_label]
            mae = all_mae[ds_label]
            rmse = all_rmse[ds_label]
            mean = all_mean[ds_label]

            var = all_var[ds_label]
            var2 = all_var2[ds_label]
            count = all_count[ds_label]
            delta = all_delta[ds_label]

            for param in params:

                model_values = model.values[param]  # type: np array
                sonde_values = sonde.values[param]  # type: np array
                if model_values is None or sonde_values is None:
                    continue

                if param == "wdir_deg":  # define dir values from u v  again  model_values, sonde_values not relevant for wind dir
                    for ix in range(timesNum):
                        if mean[param][ix] is not np.nan and model_values[ix] is not np.nan:
                            delta[param][ix] = util.to_degrees(mean["u10_ms"][ix], mean["v10_ms"][ix]) \
                                                     - util.to_degrees(model.values["u10_ms"][ix], model.values["v10_ms"][ix])
                         # print sonde_delta[param][ix]

                    for ix in range(timesNum):
                        if np.isnan(delta[param][ix]):
                            # skip assigment,
                            # in case of a single day, we will get 0 error- which means - no data , so one can not tell whether its a perfect
                            # match between model and rad or no data case
                            delta[param][ix] = 0
                        else:
                            delta[param][ix] = np.abs(util.wrap_degrees(delta[param][ix]))


                else:

                    for ix in range(timesNum):
                        if not np.isnan(model_values[ix]) and not np.isnan(mean[param][ix]):
                            delta[param][ix] = mean[param][ix] - model_values[ix]
                        else:
                           delta[param][ix] = 0

                        # second term for wind dir variance calculations:
                var2[param] += delta[param]
                var[param] += delta[param] ** 2

            # finalize computation:
            # calculate variance:
            for param in params:
                if param == "wdir_deg":
                    for ix in range(timesNum):
                        if count[param][ix] != 0.:
                            var[param][ix] = (var[param][ix] / count[param][ix] - (
                                        var2[param][ix] / count[param][ix]) ** 2) ** 0.5

                else:

                    for ix in range(timesNum):
                        if count[param][ix] != 0.:
                            var[param][ix] = (var[param][ix] / count[param][ix]) ** 0.5

    # print number of events :
    #print('number of days [wdir] = ', count["wdir_deg"], count["wdir_deg"], count["wdir_deg"])

    ########################################

    # dont display edges of interpolation results
    # suffer from low number of points
    # radiosonde dont have a lot of points at the higher levels 20-24 km
    # therefore nf=len(heights)-2
    # draw from index 2 , eg mae[draw_param][2: nf]
    #


    for draw_param in ["wdir_deg", "wvel_ms", "temp2m_k", "rh2m"]:
        nf = timesNum - 2

        xlim = param_ranges[draw_param]
        series = {}
        for ds_label in dataset_labels:
            series[ds_label] = all_values[ds_label][draw_param]


        prefix = f'surface_timeseries_{config}_Values_{start_date.strftime("%Y%m%d%H")}_{domain_label}_{draw_param}_{station.name}'

        title = f"{draw_param.upper()}, {station.name}, {start_date.strftime('%Y-%m-%d %H')}Z+{int((end_date-start_time).total_seconds()/3600)}hrs"
        print(" * " + title)
        is_angular = "wdir_deg" == draw_param
        plot_outdir = f'{outdir}/{start_date.strftime("%Y%m%d%H")}/{domain_label}/Values/'
        os.makedirs(plot_outdir, exist_ok=True)

        plot_time_series(
            Series(ref_series.xs, series, "", ["wdir_deg"]),
            plot_outdir, xlim, title, prefix)


total_plots = len(stations) * len(domain_groups) * len(time_groups)
plot_idx = 1
for tag in tags:
    for station in stations:
        for domain_group in domain_groups:
            for (start_time, end_time) in time_groups:
                print(f"Plotting ({plot_idx}/{total_plots}) {station.name} {'-'.join(domain_group)} {start_time} - {end_time}")
                create_plots(start_time, end_time, tag, "bulk", domain_group, station)
                plot_idx = plot_idx + 1
