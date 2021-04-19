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
import matplotlib.pyplot as plt
from taylor.taylorDiagram import TaylorDiagram
import scipy.stats as scstat
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
'''
all_stations = st.load_surface_stations('etc/stations.csv')
all_stations = {station.name:station for station in all_stations}
stations = []
for station_name in station_names:
    stations.append(all_stations[station_name])
time_range_groups = [
    (6,10),(10,22),(22,34),(34,48)
]

#time_range_groups = [ (x,x+3) for x in range(6,48,3)]


domain_groups = [["d04"]]
time_groups = [
            #(dt.datetime(2013, 7, 12, 18, 00), dt.datetime(2013, 7, 14, 18, 00)), \
            #(dt.datetime(2013, 8, 12, 18, 00),dt.datetime(2013, 8, 14, 18, 00)), \
            #(dt.datetime(2017, 9, 26, 18, 00),dt.datetime(2017, 9, 28, 18, 00)), \
            #(dt.datetime(2017, 11, 25, 18, 00),dt.datetime(2017, 11, 27, 18, 00)), \
            #(dt.datetime(2018, 2, 15, 18, 00),dt.datetime(2018, 2, 17, 18, 00)), \
            #(dt.datetime(2018, 4, 30, 18, 00),dt.datetime(2018, 5, 2, 18, 00)), \
            #(dt.datetime(2020, 9, 13, 18, 00),dt.datetime(2020, 9, 15, 18, 00)), \
            #(dt.datetime(2020, 9, 14, 18, 00),dt.datetime(2020, 9, 16, 18, 00)), \
            #(dt.datetime(2020, 9, 15, 18, 00),dt.datetime(2020, 9, 17, 18, 00)), \
            (dt.datetime(2016, 10, 12, 18, 00), dt.datetime(2016, 10, 15, 00, 00)),
            #(dt.datetime(2018, 4, 29, 18, 00), dt.datetime(2018, 5, 2, 18, 00))

]

tags = ['config1']

#print(obs_series)

params = ["wvel_ms", "wdir_deg", "temp2m_k", "u10_ms", "v10_ms", "rh2m"]
param_ranges = {
"wvel_ms":(0,10), "wdir_deg":(0, 120), "temp2m_k":(0,10), "rh2m": (0,50)
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



def create_plots(start_date, end_date, tag, config, domain_group, station):

    outdir = f'plots/{tag}/series/'
    os.makedirs(outdir, exist_ok=True)
    #station = sid.stations[wmoId]
    #stations = [station]

    all_datasets = tag_datasets[tag]

    domain_label = "-".join(domain_group)

    ####################################################
    # prepare datasets to extract data from them
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
    ref_series = all_datasets[sid.DATASET_LABEL].get_time_series(station, start_date, end_date, params)
    if ref_series is None:
        print(f"Missing data for {station.wmoid} {start_date}")
        return

    curr_series = {}
    for ds_label in dataset_labels:
        dataset = all_datasets[ds_label]
        series = dataset.get_time_series(station, start_date, end_date, params)

        curr_series[ds_label] = series
    all_series = [(curr_series, start_date)]



    if len(all_series) == 0:
        print("No data found")
        exit()

    ####################################################
    # prepare dictionaries for statistical properties
    times_num = len(time_range_groups)
    all_inputs = {}
    for ds_label in dataset_labels:
        ds_input = all_inputs[ds_label] = {}
        for param in params:
            param_input = ds_input[param] = {}
            for time_range_group in time_range_groups:
                param_input[time_range_group] = ([],[])

    ####################################################
    # extract input series for each dataset and param
    #print("Calculating statistics...")
    for ( profiles, curr_date) in all_series:
        surface_raw = ref_ds.get_time_series(station, start_date, end_date, params)
        for ds_label in iter(profiles):
            model = profiles[ds_label]

            surface = surface_raw.interpolate(model.xs)
            ds_input = all_inputs[ds_label]

            for param in params:
                param_input = ds_input[param]

                model_values = model.values[param]  # type: nparray
                surface_values = surface.values[param]  # type: nparray

                if model_values is None or surface_values is None:
                    continue


                for ix, xtime in enumerate(model.xs):


                    for time_ix, time_range in enumerate(time_range_groups):
                        (open_hour, close_hour) = time_range
                        time_range_min = curr_date + dt.timedelta(hours=open_hour)
                        time_range_max = curr_date + dt.timedelta(hours=close_hour)
                        if xtime >= time_range_min.timestamp()*1000 and xtime < time_range_max.timestamp()*1000:
                            (x_input, y_input) = param_input[time_range]
                            if not np.isnan(surface_values[ix]) and not np.isnan(model_values[ix]):
                                x_input.append(surface_values[ix])
                                y_input.append(model_values[ix])

    return all_inputs




total_plots = len(stations) * len(domain_groups) * len(time_groups)
plot_idx = 1
for tag in tags:
    for (start_time, end_time) in time_groups:
        for domain_group in domain_groups:
            series = {}
            for station in stations:

                print(f"Plotting ({plot_idx}/{total_plots}) {station.wmoid} {domain_group[0]} {start_time} - {end_time}")
                station_series = create_plots(start_time, end_time, tag, "bulk", domain_group, station)

                series[station] = station_series
                plot_idx = plot_idx + 1

            all_samples = {}

            model_labels = []
            for station in stations:
                station_series = series[station]
                if station_series is None:
                    continue
                for model_label in station_series.keys():
                    if model_label != "surface obs":
                        model_labels.append(model_label)
                    model_series = station_series[model_label]
                    for param in params:
                        param_series = model_series[param]
                        for time_range in time_range_groups:
                            (x_input, y_input) = time_series = param_series[time_range]
                            if len(x_input) < 2:
                                continue
                            stddev = scstat.tstd(y_input)
                            (corrcoef, p) = scstat.pearsonr(x_input, y_input)
                            key = (station.wmoid, model_label, param, time_range)
                            all_samples[key] = (stddev, corrcoef)



            for model_label in model_labels:
                for param in params:
                    for time_range in time_range_groups:

                        ################
                        # TODO: averaging station STD to mark reference line
                        std_avg = 0
                        std_count = 0
                        for i, station in enumerate(stations):
                            station_series = series[station]
                            key = (station.wmoid, "surface obs", param, time_range)
                            if not key in all_samples:
                                continue
                            (stddev, corrcoef) = all_samples[key]
                            std_avg += stddev
                            std_count += 1

                        fig = plt.figure(figsize=(6, 6))
                        fig.suptitle(f"{model_label} {param} {time_range}", size='x-large')

                        ref_std = std_avg / std_count
                        x95 = [0.05, 13.9]  # For Prcp, this is for 95th level (r = 0.195)
                        y95 = [0.0, 71.0]
                        x99 = [0.05, 19.0]  # For Prcp, this is for 99th level (r = 0.254)
                        y99 = [0.0, 70.0]
                        dia = TaylorDiagram(ref_std, fig=fig, rect=111,
                                            label='Surface Obs')

                        colors = plt.matplotlib.cm.tab20(np.linspace(0, 1, len(stations)))
                        for i, station in enumerate(stations):
                            station_series = series[station]
                            key = (station.wmoid, model_label, param, time_range)
                            if not key in all_samples:
                                continue
                            (stddev, corrcoef) = all_samples[key]
                            # Add samples to Taylor diagram
                            dia.add_sample(stddev, corrcoef, #marker='$%d$' % (i + 1)
                                           marker='o', ms=10, ls='',
                                           # mfc='k', mec='k', # B&W
                                           mfc=colors[i], mec=colors[i],  # Colors
                                           label=station.wmoid)

                        dia.ax.plot(x95, y95, color='k')
                        dia.ax.plot(x99, y99, color='k')



                        # Add samples to Taylor diagram

                        # Add RMS contours, and label them
                        contours = dia.add_contours(levels=5, colors='0.5')  # 5 levels
                        dia.ax.clabel(contours, inline=1, fontsize=10, fmt='%.1f')
                        # Tricky: ax is the polar ax (used for plots), _ax is the
                        # container (used for layout)
                        #dia._ax.set_title(domain_group)

                        # Add a figure legend and title. For loc option, place x,y tuple inside [ ].
                        # Can also use special options here:
                        # http://matplotlib.sourceforge.net/users/legend_guide.html

                        fig.legend(dia.samplePoints,
                                   [p.get_label() for p in dia.samplePoints],
                                   numpoints=1, prop=dict(size='small'), loc='upper right')

                        fig.tight_layout()

                        plt.savefig('test_taylor_4panel.png')
                        plt.show()