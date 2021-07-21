import datetime as dt
from datasets.wrf_dataset import WRFDataset
import datasets.surface_dataset as sid
import numpy as np
import scipy.stats as scst
from time_series import Series
from plot_profile import plot_bars
import os
import statistics_util as sutil
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

import pytz

def create_plots(start_date, end_date, configs, tag, domain_group, station, window):
    outdir = f'{timeseries.outdir}/{tag}/series/'
    os.makedirs(outdir, exist_ok=True)
    #station = sid.stations[wmoId]
    #stations = [station]

    all_datasets = tag_datasets[tag]

    domain_label = "-".join(domain_group)

    # wrfpld04_series = wrfpld04.get_time_series(station,  start_time, end_time,  params)
    dataset_labels = [sid.DATASET_LABEL]
    for domain in domain_group:
        for cfg in configs:
            dataset_labels.append(f"{configs[cfg]} {domain.upper()}")

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
        series = dataset.get_time_series(station, start_date, end_date, params)

        curr_series[ds_label] = series
    all_series = [(curr_series, pytz.utc.localize(start_date))]

    time_range_groups_labels = [(start_date+dt.timedelta(hours=x)).strftime("%d-%m %HZ") for (x,y) in time_range_groups]
    (_, y) = time_range_groups[-1]
    time_range_groups_labels.append((start_date+dt.timedelta(hours=y)).strftime("%d-%m %HZ"))


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

        bias = all_bias[ds_label] = {}
        mae = all_mae[ds_label] = {}
        rmse = all_rmse[ds_label] = {}
        mean = all_mean[ds_label] = {}
        var = all_var[ds_label] = {}
        var2 = all_var2[ds_label] = {}
        count = all_count[ds_label] = {}
        delta = all_delta[ds_label] = {}

        for param in params:

            bias[param] = np.zeros(times_num)
            mae[param] = np.zeros(times_num)
            rmse[param] = np.zeros(times_num)
            mean[param] = np.zeros(times_num)

            var[param] = np.zeros(times_num)
            var2[param] = np.zeros(times_num)

            count[param] = np.zeros(times_num)

    if window == 0:
        window_steps = 0
    else:
        window_steps = 2*int(window/domain_timestep[domain])+1
    window_hours = int(window / 60)
    window_tag = f"{2 * int(window_hours)}hrs avg"
    if window == 0:
        window_tag = "model steps"
    window_str = "model steps"
    if window > 0:
        window_str = f"+/-{int(window_hours)}hrs avg"

    #print("Calculating statistics...")
    for ( profiles, curr_date) in all_series:
        surface = ref_ds.get_time_series(station, start_date, end_date, params)
        for ds_label in iter(profiles):
            if ds_label.startswith("surface obs"):
                continue
            model = profiles[ds_label]


            surface = surface.interpolate(model.xs)
            if window_steps > 0:
                model = model.integrate(window_steps)

            bias = all_bias[ds_label]
            mae = all_mae[ds_label]
            rmse = all_rmse[ds_label]
            #mean = all_mean[ds_label]
            #var = all_var[ds_label]
            var2 = all_var2[ds_label]
            count = all_count[ds_label]
            #delta = all_delta[ds_label]
            mean[param] = np.zeros(times_num)
            for param in params:

                (xs, delta, surface_values, model_values) = sutil.get_delta_series(model, surface, param)
                if delta is None:
                    continue

                ################
                # debug delta values
                '''
                datetimes = [dt.datetime.utcfromtimestamp(millis / 1000) for millis in xs]

                xticks = [(start_date + dt.timedelta(hours=x)) for (x, y) in time_range_groups]
                (_, y) = time_range_groups[-1]
                xticks.append((start_date + dt.timedelta(hours=y)))
                xlabels = [tick.strftime("%d-%m %H:%MZ") for tick in xticks]
                plt.plot(datetimes, delta)
                plt.xticks(xticks, xlabels, rotation=30)
                plt.gca().xaxis.grid(True, linestyle='--')
                plt.gca().yaxis.grid(True, linestyle='--')
                plt.show()
                '''


                deltas = {}
                for ix, xtime in enumerate(xs):

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
                        if not range_ix in deltas:
                            deltas[range_ix] = []
                        deltas[range_ix].append(delta[ix])

                for range_ix, (open_hour, close_hour) in enumerate(time_range_groups):

                    if count[param][range_ix] >= 5:
                        bias[param][range_ix] /= count[param][range_ix]
                        mae[param][range_ix] /= count[param][range_ix]
                        rmse[param][range_ix] = (rmse[param][range_ix] / count[param][range_ix]) ** 0.5
                        if param == "wdir_deg":
                            var2[param][range_ix] = scst.circstd(deltas[range_ix], low=0, high=360, nan_policy='omit')
                        else:
                            var2[param][range_ix] = np.nanstd(deltas[range_ix])
                #if not np.isnan(mean["u10_ms"][ix]):
                    #mean["wdir_deg"][ix] = util.to_degrees(mean["u10_ms"][ix], mean["v10_ms"][ix])

    # completed mean bias ame rmse calculations
    # print sonde_mean["wdir_deg"]
    all_values = {
        'Bias': all_bias,
        'MAE': all_mae,
        'RMSE': all_rmse
    }
    for metrics_name in all_values.keys():
        metrics_values = all_values[metrics_name]
        for draw_param in ["wvel_ms", "wdir_deg", "temp2m_k", "rh2m"]:
            nf = times_num - 2

            series = {}
            errors = {}

            for ds_label in dataset_labels:
                if not ds_label.startswith("surface obs"):
                    series[ds_label] = metrics_values[ds_label][draw_param]
                    if metrics_name == "Bias":
                        errors[ds_label] = all_var2[ds_label][draw_param]

            cfg = 'All'
            prefix = f'surface_timeseries_{cfg}_{metrics_name}_{start_date.strftime("%Y%m%d%H")}_{domain_label}_{draw_param}_{station.name}_{window_tag}'
            title = f"{draw_param.upper()} {metrics_name}, {cfg}, {station.name}, {domain_label}"
            if window_hours != 0:
                title += f', {window_str}'
            plot_outdir = f'{outdir}/{start_date.strftime("%Y%m%d%H")}/{domain_label}/{metrics_name}'
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
    total_plots = len(tags)*len(stations) * len(domain_groups) * len(time_groups) * len(windows)
    plot_idx = 1

    for tag in tags:
        for station in stations:
            for domain_group in domain_groups:
                for (start_time, end_time) in time_groups:
                    for window in windows:
                        print(f"Plotting ({plot_idx}/{total_plots}) {station.wmoid} {domain_group[0]} {start_time} - {end_time}")
                        create_plots(start_time, end_time, configs, tag, domain_group, station, window)
                        plot_idx = plot_idx + 1


if __name__ == "__main__":
    generate(configs, timeseries.stations, timeseries.domain_groups, timeseries.time_groups)
