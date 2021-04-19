import pandas as pd

import datetime as dt
from datasets.wrf_dataset import WRFDataset
import datasets.surface_dataset as sid
from datasets.surface_israel_dataset import IsraelSurfaceDataset
import datasets.util as util
import numpy as np
from time_series import Series
from plot_profile import plot_bars
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
'''
station_names = [
    "Ein Karmel",
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
]'''
all_stations = st.load_surface_stations_map('etc/stations_and_neighbors.csv')
stations = []
for station_name in station_names:
    stations.append(all_stations[station_name])
time_range_groups = [
    (6,10),(10,22),(22,34),(34,48)
]

#time_range_groups = [ (x,x+3) for x in range(6,48,3)]


domain_groups = [["d04"]]
#domain_groups = [["d01"], ["d02"], ["d03", "d04"]]
#dt.datetime(2013, 7, 12, 18, 00), dt.datetime(2013, 7, 14, 18, 00)
time_groups = [
            (dt.datetime(2013, 7, 12, 18, 00), dt.datetime(2013, 7, 14, 18, 00)), \
            (dt.datetime(2017, 9, 26, 18, 00),dt.datetime(2017, 9, 28, 18, 00)), \
            (dt.datetime(2017, 11, 25, 18, 00),dt.datetime(2017, 11, 27, 18, 00)), \
            (dt.datetime(2018, 2, 15, 18, 00),dt.datetime(2018, 2, 17, 18, 00)), \
            (dt.datetime(2018, 4, 30, 18, 00),dt.datetime(2018, 5, 2, 18, 00)), \
            (dt.datetime(2020, 9, 13, 18, 00),dt.datetime(2020, 9, 15, 18, 00)), \
            (dt.datetime(2020, 9, 14, 18, 00),dt.datetime(2020, 9, 16, 18, 00)), \
            (dt.datetime(2020, 9, 15, 18, 00),dt.datetime(2020, 9, 17, 18, 00)), \
            (dt.datetime(2016, 10, 12, 18, 00), dt.datetime(2016, 10, 15, 00, 00)),
            (dt.datetime(2018, 4, 29, 18, 00), dt.datetime(2018, 5, 2, 18, 00))

]

tags = ['config1']

#print(obs_series)

params = ["wvel_ms", "wdir_deg", "temp2m_k", "u10_ms", "v10_ms", "rh2m"]
param_ranges = {
"wvel_ms":(0,10), "wdir_deg":(0, 150), "temp2m_k":(0,10), "rh2m": (0,50)
}

surface_ds = sid.SurfaceDataset(sid.surface_archive_dir)
configs = ['bulk_sst'] #
tag_datasets = {}

for tag in tags:
    domain_datasets = {}
    base_wrf_dir = r"E:\meteo\urban-wrf\wrfout\\"
    for domain in ["d01", "d02", "d03", "d04"]:
        for cfg in configs:
            domain_datasets [f"{cfg} {domain.upper()}"] = WRFDataset(f"{base_wrf_dir}\\{cfg}", domain)
    domain_datasets[sid.DATASET_LABEL] = surface_ds
    tag_datasets[tag] = domain_datasets



def create_plots(start_date, end_date, tag, config, domain_group, stations):

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


    ref_series = {}
    ####################################################
    # caching all relevant data:
    for station in stations:
        ref_series[station.wmoid] = all_datasets[sid.DATASET_LABEL].get_time_series(station, start_date, end_date, params)

    if ref_series is None:
        print(f"Missing data for {station.wmoid} {start_date}")
        return

    curr_series = {}
    for station in stations:
        for ds_label in dataset_labels:
            dataset = all_datasets[ds_label]
            series = dataset.get_time_series(station, start_date, end_date, params)

            curr_series[f'{ds_label}_{station.wmoid}'] = series
    all_series = [(curr_series, start_date)]



    if len(all_series) == 0:
        print("No data found")
        exit()

    times_num = len(time_range_groups)
    all_bias = {}
    all_mae = {}
    all_rmse = {}
    all_mean = {}
    all_var = {}
    all_var2 = {}
    all_count = {}  # at each z level number of events  when model and measurement exist
    all_delta = {}
    for ds_label in dataset_labels:
        for station in stations:
            point_label = f'{ds_label}_{station.wmoid}'
            bias = all_bias[point_label] = {}
            mae = all_mae[point_label] = {}
            rmse = all_rmse[point_label] = {}
            mean = all_mean[point_label] = {}
            var = all_var[point_label] = {}
            var2 = all_var2[point_label] = {}
            count = all_count[point_label] = {}
            delta = all_delta[point_label] = {}

            for param in params:

                bias[param] = np.zeros(times_num)
                mae[param] = np.zeros(times_num)
                rmse[param] = np.zeros(times_num)
                mean[param] = np.zeros(times_num)

                var[param] = np.zeros(times_num)
                var2[param] = np.zeros(times_num)

                count[param] = np.zeros(times_num)
                delta[param] = np.zeros(times_num)
                delta[param][:] = np.nan

    #print("Calculating statistics...")
    for station in stations:
        for ( profiles, curr_date) in all_series:
            surface_raw = ref_series[station.wmoid]
            if surface_raw is None:
                continue
                #print(station.wmoid)
            for ds_label in dataset_labels:
                point_label = f'{ds_label}_{station.wmoid}'
                model = profiles[point_label]

                surface = surface_raw.interpolate(model.xs)

                bias = all_bias[point_label]
                mae = all_mae[point_label]
                rmse = all_rmse[point_label]
                mean = all_mean[point_label]
                var = all_var[point_label]
                var2 = all_var2[point_label]
                count = all_count[point_label]
                delta = all_delta[point_label]

                for param in params:
                    model_values = model.values[param]  # type: nparray
                    surface_values = surface.values[param]  # type: nparray

                    if model_values is None or surface_values is None:
                        continue

                    delta = np.zeros(len(model.xs))
                    delta[:] = np.nan
                    for ix, xtime in enumerate(model.xs):

                        if np.isnan(model_values[ix]) or np.isnan(surface_values[ix]):
                            delta[ix] = 0  # delta = 0 , do not increase number of events
                            continue
                        if param == "wdir_deg":
                            su = surface.values["u10_ms"][ix]
                            sv = surface.values["v10_ms"][ix]
                            if (su**2+sv**2) >= 1:
                                mu = model.values["u10_ms"][ix]
                                mv = model.values["v10_ms"][ix]
                                ds = util.to_degrees(su, sv)
                                dm = util.to_degrees(mu, mv)
                                diff = ds - dm
                                source_angle_diff = (diff + 180) % 360 - 180
                                delta[ix] = source_angle_diff

                        elif param == "wvel_ms":
                            su = surface.values["u10_ms"][ix]
                            sv = surface.values["v10_ms"][ix]
                            if (su**2+sv**2) >= 1:
                                delta[ix] = surface_values[ix] - model_values[ix]
                        else:
                            delta[ix] = surface_values[ix] - model_values[ix]

                        if np.isnan(delta[ix]):
                            continue

                        range_ix = None
                        for time_ix, (open_hour, close_hour) in enumerate(time_range_groups):
                            time_range_min = curr_date + dt.timedelta(hours=open_hour)
                            time_range_max = curr_date + dt.timedelta(hours=close_hour)
                            if xtime >= time_range_min.timestamp()*1000 and xtime < time_range_max.timestamp()*1000:
                                range_ix = time_ix

                        if range_ix is not None:
                            count[param][range_ix] += 1

                            bias[param][range_ix] += delta[ix]

                            mae[param][range_ix] += abs(delta[ix])
                            rmse[param][range_ix] += delta[ix] ** 2
                        #mean[param][ix] += model_values[ix]

                    # print count[param]



                for ix in range(times_num):

                    for param in params:
                        if param != "wdir_deg":
                                if count[param][ix] != 0:
                                    mean[param][ix] /= count[param][ix]
                        if count[param][ix] != 0:
                            bias[param][ix] /= count[param][ix]
                            mae[param][ix] /= count[param][ix]
                            rmse[param][ix] = (rmse[param][ix] / count[param][ix]) ** 0.5

                    #if not np.isnan(mean["u10_ms"][ix]):
                        #mean["wdir_deg"][ix] = util.to_degrees(mean["u10_ms"][ix], mean["v10_ms"][ix])

    # completed mean bias ame rmse calculations
    # print sonde_mean["wdir_deg"]
    all_values = {
        'Bias': all_bias,
        'MAE': all_mae,
        'RMSE': all_rmse
    }
    time_range_groups_labels = [start_date+dt.timedelta(hours=x) for (x,y) in time_range_groups]
    (_, y) = time_range_groups[-1]
    time_range_groups_labels.append(start_date+dt.timedelta(hours=y))
    for metrics_name in all_values.keys():
        metrics_values = all_values[metrics_name]
        for draw_param in ["wdir_deg", "wvel_ms", "temp2m_k", "rh2m"]:


            series = {}

            for station in stations:
                for ds_label in dataset_labels:
                    point_label = f'{ds_label}_{station.wmoid}'
                    if not ds_label.startswith("surface obs"):
                        series[station.wmoid] = metrics_values[point_label][draw_param]


            prefix = f'surface_timeseries_{config}_{metrics_name}_{start_date.strftime("%Y%m%d%H")}_{domain_label}_{draw_param}_All Stations'

            title = f"{draw_param.upper()} {domain_label} {metrics_name}, {config}, {start_date.strftime('%Y-%m-%d %H')}Z"
            print(" * " + title)
            is_angular = "wdir_deg" == draw_param
            plot_outdir = f'{outdir}/{start_date.strftime("%Y%m%d%H")}/{domain_label}/{metrics_name}'
            os.makedirs(plot_outdir, exist_ok=True)
            (ymin,ymax) = param_ranges[draw_param]
            if metrics_name == "Bias":
                ylim = (-ymax*2/3, ymax*2/3)
            else:
                ylim = (0, ymax)

            plot_bars(
                Series(time_range_groups_labels, series, "", ["wdir_deg"]),
                plot_outdir, ylim=ylim, title=title, prefix=prefix, side_legend=True)


total_plots = len(domain_groups) * len(time_groups)
plot_idx = 1
for tag in tags:
    for domain_group in domain_groups:
        for (start_time, end_time) in time_groups:
            print(f"Plotting ({plot_idx}/{total_plots}) {domain_group[0]} {start_time} - {end_time}")
            create_plots(start_time, end_time, tag, "bulk", domain_group, stations)
            plot_idx = plot_idx + 1