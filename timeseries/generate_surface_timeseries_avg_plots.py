from datasets.wrf_dataset import WRFDataset
import datasets.surface_dataset as sid
from plot_profile import plot_time_series
import os
import sys
import timeseries.timeseries_cfg as timeseries

tags = timeseries.tags
params = timeseries.params
param_ranges={"wvel_ms":(0, 15), "wdir_deg":(0, 360), "temp2m_k":(280, 310), "u_ms":(0, 50), "v_ms":(0,50), "rh2m":(0,100)}

configs = timeseries.configs
time_range_groups = timeseries.time_range_groups
domain_timestep = timeseries.domain_timestep



surface_ds = sid.SurfaceDataset(sid.surface_archive_dir)

tag_datasets = {}

for tag in tags:
    domain_datasets = {sid.DATASET_LABEL: surface_ds}
    base_wrf_dir = r"E:\meteo\urban-wrf\wrfout\\"
    for domain in ["d01", "d02", "d03", "d04"]:
        for cfg in configs:
            domain_datasets [f"{configs[cfg]} {domain.upper()}"] = WRFDataset(f"{base_wrf_dir}\\{cfg}", domain)
    tag_datasets[tag] = domain_datasets


def create_plots(start_date, end_date, tag, configs, domain_group, station, window):

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

    #print("Caching profiles...")

    ref_series = all_datasets[sid.DATASET_LABEL].get_time_series(station, start_date, end_date, params)
    if ref_series is None:
        print(f"Missing data for {station.wmoid} {start_date}")
        return
    timesNum = len(ref_series.xs)



    curr_series = {}
    for ds_label in dataset_labels:
        dataset = all_datasets[ds_label]
        if ds_label.startswith(sid.DATASET_LABEL):
            curr_series[ds_label] = dataset.get_time_series(station, start_date, end_date, params)
            continue

        if window == 0:
            window_steps = 0
        else:
            window_steps = 2 * int(window / domain_timestep[domain]) + 1
        window_hours = int(window / 60)
        window_tag = f"{2 * int(window_hours)}hrs avg"
        if window == 0:
            window_tag = "model steps"
        window_str = ""
        if window > 0:
            window_str = f"+/-{int(window_hours)}hrs avg"

        series_tag = f'{ds_label} {window_str}'
        series = dataset.get_time_series(station, start_date, end_date, params)

        if window_steps != 0:
            series = series.integrate(window_steps)

        curr_series[series_tag] = series
    all_series = [(curr_series, start_date)]


    if len(all_series) == 0:
        print("No data found")
        exit()


    all_values = {}

    for ds_label in curr_series:
        all_values[ds_label] = {}


    #print("Calculating statistics...")
    for ( profiles, curr_date) in all_series:
        surface = ref_ds.get_time_series(station, start_date, end_date, params)
        for ds_label in iter(profiles):
            model = profiles[ds_label]
            values = all_values[ds_label]
            for param in params:
                values[param] = (model.values[param], model.xs)

    for draw_param in ["wdir_deg", "wvel_ms", "temp2m_k", "rh2m"]:
        nf = timesNum - 2

        xlim = param_ranges[draw_param]
        series = {}
        for ds_label in curr_series:
            series[ds_label] = all_values[ds_label][draw_param]

        cfg = 'All'
        prefix = f'surface_timeseries_{cfg}_Values_{start_date.strftime("%Y%m%d%H")}_{domain_label}_{draw_param}_{station.name}_{window_tag}'

        title = f"{draw_param.upper()}, {cfg}, {station.name}, {start_date.strftime('%Y-%m-%d %H')}Z+{int((end_date-start_date).total_seconds()/3600)}hrs"
        print(" * " + title)
        is_angular = "wdir_deg" == draw_param
        plot_outdir = f'{outdir}/{start_date.strftime("%Y%m%d%H")}/{domain_label}/Values/'
        os.makedirs(plot_outdir, exist_ok=True)

        plot_time_series( ref_series.xs, series,
            plot_outdir, xlim, title, prefix)

windows = [0, 60, 120]
def generate(configs, stations, domain_groups, time_groups):

    total_plots = len(tags)*len(stations) * len(domain_groups) * len(time_groups)*len(windows)
    plot_idx = 1


    for tag in tags:
            for station in stations:
                for domain_group in domain_groups:
                    for (start_time, end_time) in time_groups:
                        for window in windows:
                            print(f"Plotting ({plot_idx}/{total_plots}) {station.name} {'-'.join(domain_group)} {start_time} - {end_time}")
                            try:
                                create_plots(start_time, end_time, tag, configs, domain_group, station, window)
                            except:
                                print( "Unexpected error:", sys.exc_info()[0])
                            plot_idx = plot_idx + 1

if __name__ == "__main__":
    generate(configs, timeseries.stations, timeseries.domain_groups, timeseries.time_groups)
