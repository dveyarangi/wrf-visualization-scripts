import datetime as dt
from datasets.wrf_dataset import WRFDataset
import datasets.surface_dataset as sid
import numpy as np
import astropy.stats as asst
from time_series import Series
from plot_profile import plot_bars
import os
import timeseries.timeseries_cfg as timeseries

tags = timeseries.tags
params = timeseries.params
param_ranges = timeseries.param_ranges
configs = timeseries.configs
time_range_groups = timeseries.time_range_groups
domain_timestep = timeseries.domain_timestep



surface_ds = sid.SurfaceDataset(sid.surface_archive_dir)

#configs = {'slucm':'slucm'}
tag_datasets = {}

for tag in tags:
    domain_datasets = {}
    base_wrf_dir = r"E:\meteo\urban-wrf\wrfout\\"
    for domain in ["d01", "d02", "d03", "d04"]:
        for cfg in configs:
            domain_datasets [f"{configs[cfg]} {domain.upper()}"] = WRFDataset(f"{base_wrf_dir}\\{cfg}", domain)
    domain_datasets[sid.DATASET_LABEL] = surface_ds
    tag_datasets[tag] = domain_datasets

import pytz

def create_plots(start_date, end_date, tag, cfg, domain_group, station, window):
    outdir = f'{timeseries.outdir}/{tag}/series/'
    os.makedirs(outdir, exist_ok=True)
    #station = sid.stations[wmoId]
    #stations = [station]

    all_datasets = tag_datasets[tag]

    domain_label = "-".join(domain_group)

    # wrfpld04_series = wrfpld04.get_time_series(station,  start_time, end_time,  params)
    dataset_labels = [sid.DATASET_LABEL]
    for domain in domain_group:
        dataset_labels.append(f"{cfg} {domain.upper()}")

    datasets = []

    for ds_label in dataset_labels:
        datasets.append(all_datasets[ds_label])

    ref_ds = surface_ds


    ####################################################
    # caching all relevant data:
    ref_series = all_datasets[sid.DATASET_LABEL].get_time_series(station, start_date, end_date, params)
    if ref_series is None:
        print(f"Missing data for {station.wmoid} {start_date}")
        return

    curr_series = {}
    for ds_label in dataset_labels:
        dataset = all_datasets[ds_label]
        if ds_label.startswith(sid.DATASET_LABEL):
            curr_series[ds_label] = dataset.get_time_series(station, start_date, end_date, params)
            continue
        for window in windows:
            if window == 0:
                window_steps = 0
            else:
                window_steps = int(window / domain_timestep[domain])
            window_hours = int(window / 60)
            window_tag = "model steps"
            window_str = "model steps"
            if window != 0:
                window_str = f"{int(window_hours)}hrs offset"
                if window > 0: window_str = f'+{window_str}'

            series_tag = f'{ds_label} {window_str}'
            series = dataset.get_time_series(station, start_date, end_date, params)

            if window_steps != 0:
                series = series.shift(window_steps)

            curr_series[series_tag] = series


    all_series = [(curr_series, pytz.utc.localize(start_date))]

    time_range_groups_labels = [(start_date+dt.timedelta(hours=x)).strftime("%d-%m %HZ") for (x,y) in time_range_groups]
    (_, y) = time_range_groups[-1]
    time_range_groups_labels.append((start_date+dt.timedelta(hours=y)).strftime("%d-%m %HZ"))


    if len(all_series) == 0:
        print("No data found")
        exit()

    times_num = len(time_range_groups)

    all_corr = {}

    all_count = {}
    for ds_label in curr_series:
        corr = all_corr[ds_label] = {}
        count = all_count[ds_label] = {}
        for param in params:
            corr[param] = np.zeros(times_num)
            count[param] = np.zeros(times_num)



    #print("Calculating statistics...")
    for ( profiles, curr_date) in all_series:
        surface = ref_ds.get_time_series(station, start_date, end_date, params)
        for ds_label in iter(profiles):
            if ds_label.startswith("surface obs"):
                continue

            corr = all_corr[ds_label]
            count = all_count[ds_label]
            model = profiles[ds_label]

            xs = model.xs
            surface = surface.interpolate(xs)


            for param in params:

                # group by time windows
                window_obs = {}
                window_preds = {}
                for ix, xtime in enumerate(xs):

                    if np.isnan(surface.values[param][ix]) or np.isnan(model.values[param][ix]):
                        continue

                    range_ix = None

                    for time_ix, (open_hour, close_hour) in enumerate(time_range_groups):
                        time_range_min = curr_date + dt.timedelta(hours=open_hour)
                        time_range_max = curr_date + dt.timedelta(hours=close_hour)
                        if xtime >= time_range_min.timestamp()*1000 and xtime < time_range_max.timestamp()*1000:
                            range_ix = time_ix

                    if range_ix is not None:
                        count[param][range_ix] += 1

                        if not range_ix in window_obs:
                            window_obs[range_ix] = []
                            window_preds[range_ix] = []

                        window_obs[range_ix].append(surface.values[param][ix])
                        window_preds[range_ix].append(model.values[param][ix])

                for range_ix, (open_hour, close_hour) in enumerate(time_range_groups):

                    if count[param][range_ix] == 0:
                        continue

                    s = np.array(window_obs[range_ix])
                    m = np.array(window_preds[range_ix])

                    if param == "wdir_deg":
                        corr_coef = asst.circcorrcoef(np.deg2rad(m), np.deg2rad(s))
                    else:
                        corr_coef = np.corrcoef(m, s)[0,1]
                    corr[param][range_ix] = corr_coef

    # completed mean bias ame rmse calculations
    # print sonde_mean["wdir_deg"]
    all_values = {
        'Pearson': all_corr,
    }
    for metrics_name in all_values.keys():
        metrics_values = all_values[metrics_name]
        for draw_param in ["wvel_ms", "wdir_deg", "temp2m_k", "rh2m"]:
            nf = times_num - 2

            series = {}

            for ds_label in curr_series:
                if not ds_label.startswith("surface obs"):
                    series[ds_label] = metrics_values[ds_label][draw_param]

            prefix = f'surface_timeseries_{cfg}_{metrics_name}_{start_date.strftime("%Y%m%d%H")}_{domain_label}_{draw_param}_{station.name}_{window_tag}'


            title = f"{draw_param.upper()} {metrics_name}, {cfg}, {station.name}, {domain_label}"
            print(" * " + title)
            is_angular = "wdir_deg" == draw_param
            plot_outdir = f'{outdir}/{start_date.strftime("%Y%m%d%H")}/{domain_label}/{metrics_name}'
            os.makedirs(plot_outdir, exist_ok=True)
            ylim = (-0.25, 1)
            errors = None
            plot_bars(
                Series(time_range_groups_labels, series, "", ["wdir_deg"]),
                plot_outdir, errors=errors, ylim=ylim, title=title, prefix=prefix)

windows = [-60, 0, 60 ]
def generate(configs, stations, domain_groups, time_groups):

    total_plots = len(tags)*len(configs)*len(stations) * len(domain_groups) * len(time_groups)
    plot_idx = 1

    for tag in tags:
        for cfg in configs:
            for station in stations:
                for domain_group in domain_groups:
                    for (start_time, end_time) in time_groups:
                        print(f"Plotting ({plot_idx}/{total_plots}) {station.wmoid} {domain_group[0]} {start_time} - {end_time}")
                        create_plots(start_time, end_time, tag, configs[cfg], domain_group, station, windows)
                        plot_idx = plot_idx + 1
