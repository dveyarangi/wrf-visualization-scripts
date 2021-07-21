import datetime as dt
from datasets.wrf_dataset import WRFDataset
import datasets.surface_dataset as sid
import numpy as np
from time_series import Series
from plot_profile import plot_bars
import os
import statistics_util as sutil
import scipy.stats as scst
import timeseries.timeseries_cfg as timeseries

tags = timeseries.tags
params = timeseries.params
param_ranges = timeseries.param_ranges
configs = timeseries.configs
time_range_groups = timeseries.time_range_groups
domain_timestep = timeseries.domain_timestep


surface_ds = sid.SurfaceDataset(sid.surface_archive_dir)
tag_datasets = {}

for tag in tags:
    domain_datasets = {}
    base_wrf_dir = r"E:\meteo\urban-wrf\wrfout\\"
    for domain in ["d01", "d02", "d03", "d04"]:
        for cfg in configs:
            domain_datasets [f"{configs[cfg]} {domain.upper()}"] = WRFDataset(f"{base_wrf_dir}\\{cfg}", domain)
    domain_datasets[sid.DATASET_LABEL] = surface_ds
    tag_datasets[tag] = domain_datasets



def create_plots(time_groups, tag, cfg, domain_group, stations, window):

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


    ref_series = {}
    ####################################################
    # caching all relevant data:
    for station in stations:
        for (start_time, end_time) in time_groups:
            ref_series[f'{station.wmoid}_{start_time}_{end_time}'] = all_datasets[sid.DATASET_LABEL].get_time_series(station, start_time, end_time, params)

    if ref_series is None:
        print(f"Missing data for {station.wmoid}")
        return

    if window == 0:
        window_steps = 0
    else:
        window_steps = 2 * int(window / domain_timestep[domain]) + 1
    window_hours = int(window / 60)
    window_tag = f"{2 * int(window_hours)}hrs avg"
    if window == 0:
        window_tag = "model steps"
    window_str = "model steps"
    if window > 0:
        window_str = f"+/-{int(window_hours)}hrs avg"

    curr_series = {}
    for station in stations:
        for ds_label in dataset_labels:
            for (start_time, end_time) in time_groups:
                series_label = f'{ds_label}_{station}_{start_time}_{end_time}'
                dataset = all_datasets[ds_label]
                series = dataset.get_time_series(station, start_time, end_time, params)

                curr_series[series_label] = series

    times_num = len(time_range_groups)
    all_bias = {}
    all_mae = {}
    all_rmse = {}
    all_var2 = {}
    all_count = {}  # at each z level number of events  when model and measurement exist
    all_deltas = {}


    for ds_label in dataset_labels:
        bias = all_bias[ds_label] = {}
        mae = all_mae[ds_label] = {}
        rmse = all_rmse[ds_label] = {}
        var2 = all_var2[ds_label] = {}
        count = all_count[ds_label] = {}
        deltas = all_deltas[ds_label] = {}
        for param in params:
            bias[param] = np.zeros(times_num)
            mae[param] = np.zeros(times_num)
            rmse[param] = np.zeros(times_num)
            var2[param] = np.zeros(times_num)
            count[param] = np.zeros(times_num)
            deltas[param] = {i:[] for i in range(times_num)}

    #print("Calculating statistics...")
    for station in stations:
        for (start_time, end_time) in time_groups:

            surface_raw = ref_series[f'{station.wmoid}_{start_time}_{end_time}']
            if surface_raw is None:
                continue
                #print(station.wmoid)
            for ds_label in dataset_labels:
                series_label = f'{ds_label}_{station}_{start_time}_{end_time}'
                point_label = f'{ds_label}'
                model = curr_series[series_label]

                surface = surface_raw.interpolate(model.xs)
                if window_steps > 0:
                    model = model.integrate(window_steps)

                bias = all_bias[point_label]
                mae = all_mae[point_label]
                rmse = all_rmse[point_label]

                count = all_count[point_label]
                deltas = all_deltas[point_label]
                for param in params:
                    (xs, delta, surface_values, model_values) = sutil.get_delta_series(model, surface, param)
                    if delta is None:
                        continue
                    for ix, xtime in enumerate(model.xs):

                        if np.isnan(delta[ix]):
                            continue

                        range_ix = None
                        for time_ix, (open_hour, close_hour) in enumerate(time_range_groups):
                            time_range_min = start_time + dt.timedelta(hours=open_hour)
                            time_range_max = start_time + dt.timedelta(hours=close_hour)
                            if xtime >= time_range_min.timestamp()*1000 and xtime < time_range_max.timestamp()*1000:
                                range_ix = time_ix

                        if range_ix is not None:
                            count[param][range_ix] += 1

                            bias[param][range_ix] += delta[ix]

                            mae[param][range_ix] += abs(delta[ix])
                            rmse[param][range_ix] += delta[ix] ** 2

                            deltas[param][range_ix].append(delta[ix])
                        #mean[param][ix] += model_values[ix]

                    # print count[param]

    for ds_label in dataset_labels:
        point_label = f'{ds_label}'
        bias = all_bias[point_label]
        mae = all_mae[point_label]
        rmse = all_rmse[point_label]
        var2 = all_var2[point_label]
        count = all_count[point_label]
        deltas = all_deltas[point_label]
        for ix in range(times_num):

            for param in params:
                if count[param][ix] >= 5:
                    bias[param][ix] /= count[param][ix]
                    mae[param][ix] /= count[param][ix]
                    rmse[param][ix] = (rmse[param][ix] / count[param][ix]) ** 0.5

                    if param == "wdir_deg":
                        var2[param][ix] = scst.circstd(deltas[param][ix], low=0, high=360, nan_policy='omit')
                    else:
                        var2[param][ix] = np.nanstd(deltas[param][ix])
                    #if not np.isnan(mean["u10_ms"][ix]):
                        #mean["wdir_deg"][ix] = util.to_degrees(mean["u10_ms"][ix], mean["v10_ms"][ix])

    # completed mean bias ame rmse calculations
    # print sonde_mean["wdir_deg"]
    all_values = {
        'Bias': all_bias,
        'MAE': all_mae,
        'RMSE': all_rmse
    }
    time_range_groups_labels = [f'Day {int((x-6)/24)+1} {(x-6)%24:02}Z' for (x,y) in time_range_groups]
    (_, y) = time_range_groups[-1]
    time_range_groups_labels.append(f'Day {int((y-6)/24)+1} {(y-6)%24:02}Z')
    for metrics_name in all_values.keys():
        metrics_values = all_values[metrics_name]
        for draw_param in ["wdir_deg", "wvel_ms", "temp2m_k", "rh2m"]:


            series = {}

            for ds_label in dataset_labels:
                if not ds_label.startswith("surface obs"):
                    series[ds_label] = metrics_values[ds_label][draw_param]
                    if metrics_name == "Bias":
                        errors = all_var2[ds_label][draw_param]
                    else:
                        errors= None

            prefix = f'surface_timeseries_{cfg}_{metrics_name}_All Events_{domain_label}_{draw_param}_All Stations Avg_{window_tag}'

            title = f"{draw_param.upper()} {domain_label} {metrics_name}, {cfg}, All Events&StationsAvg."
            if window_steps != 0:
                title += f', {window_str}'
            print(" * " + title)
            is_angular = "wdir_deg" == draw_param
            plot_outdir = f'{outdir}/All Events/{domain_label}/{metrics_name}'
            os.makedirs(plot_outdir, exist_ok=True)
            (ymin,ymax) = param_ranges[draw_param]
            if metrics_name == "Bias":
                ylim = (-ymax*2/3, ymax*2/3)
            else:
                ylim = (0, ymax)

            plot_bars(
                Series(time_range_groups_labels, series, "", ["wdir_deg"]),
                plot_outdir, errors=errors, ylim=ylim, title=title, prefix=prefix)

def generate(configs, stations, domain_groups, time_groups):
    windows = [0, 60, 120]

    total_plots = len(configs)*len(windows)*len(domain_groups)
    plot_idx = 1
    for tag in tags:
        for cfg in configs:
            for domain_group in domain_groups:
                for window in windows:
                    print(f"Plotting ({plot_idx}/{total_plots}) {cfg} w{window} {domain_group[0]}...")
                    create_plots(time_groups, tag, configs[cfg], domain_group, stations, window)
                    plot_idx = plot_idx + 1