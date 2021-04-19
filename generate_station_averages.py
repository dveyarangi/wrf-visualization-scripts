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
import calendar
import os
import scipy.stats as scst
import matplotlib.pyplot as plt

from plot_profile import plot_line_and_errors

outdir = f'plots/stations/'
os.makedirs(outdir, exist_ok=True)
start_time = dt.datetime(2013, 1, 1, 00, 00)
end_time = dt.datetime(2020, 12, 31, 23, 00)

station_names = [ \
    'Afek', \
    'Ein Karmel', \
    'Haifa Refineries', \
    'Haifa Technion', \
    'Haifa University', \
    'Shavei Zion', \
    'Nesher', 'K.HAIM-REGAVIM', 'K.Hasidim', 'K.Bnyamin', 'K.Tivon', 'K.Yam', 'Shprinzak', 'N.Shanan', 'Ahuza',
     'K.Bialik-Ofarim', 'IGUD', 'K.Ata']

#station_names = ['Ahuza']
all_stations = st.load_surface_stations_map('etc/stations_and_neighbors.csv')
stations = [all_stations[name] for name in station_names]



params = ["wdir_deg", "wvel_ms", "temp2m_k", "rh2m"]
param_ranges={"wvel_ms":(0, 15), "wdir_deg":(0, 360), "temp2m_k":(280, 310), "u_ms":(0, 50), "v_ms":(0,50), "rh2m":(0,100)}
valid_values_max = {"wvel_ms":20, "wdir_deg":360.01, "temp2m_k":50, "rh2m":100.01}

surface_ds = sid.SurfaceDataset(sid.surface_archive_dir)

years = range(2013,2021)
months = range(1,13)
hours = range(0, 24)

total_count = len(params) * len(stations) * len(months)
plot_count = 1
for param in params:
    averages_grid = np.zeros((len(stations), len(months), len(hours)))
    for midx, month in enumerate(months):
        for sidx, station in enumerate(stations):

            month_name = calendar.month_name[month]
            print(f"Creating plot ({plot_count}/{total_count}) for {param}, {station.wmoid}, {month_name}", end="")
            plot_count += 1
            hourly_mean = np.zeros(len(hours))
            hourly_std = np.zeros(len(hours))
            for hidx, hour in enumerate(hours):
                count = 0
                year_series = {}
                values = np.zeros((0))
                for year in years:
                    for day in range(1, calendar.monthrange(year, month)[-1]+1):
                        start_time = dt.datetime(year, month, day, hour, 00)
                        end_time = dt.datetime(year, month, day, hour, 59)
                        series = surface_ds.get_time_series(station, start_time, end_time, params)
                        if series is None:
                            continue
                        array = series.values[param]
                        values = np.append(values, array)

                values = values[np.logical_not(np.isnan(values))]

                if param == "wdir_deg":
                    mean = scst.circmean(values, low=0, high=360, nan_policy='omit')
                    std = scst.circstd(values, low=0, high=360, nan_policy='omit')
                else:
                    mean = np.nanmean(values)
                    std = np.nanstd(values)

                hourly_mean[hidx] = mean
                hourly_std[hidx] = std

                print(".", end="")

            print("done!")
            title = f"{param} Average for {station.wmoid}, {month_name} ";
            prefix = f"station_average_{param}_{station.wmoid}_{month_name}";
            colors = plt.get_cmap("tab20").colors
            color = colors[sidx * 3 % len(colors)]
            #plot_line_and_errors(hourly_mean, hourly_std, hours, outdir, color=color, , title=title, prefix=prefix)
            fig, ax = plt.subplots()

            #xticks = np.arange(len(series.xs)-1)
            ax.errorbar(hours, hourly_mean, hourly_std, linewidth=3, elinewidth=2, ecolor='grey', capsize=4)
           #ax.bar(xlabels, series, yerr=error_series, alpha=0.5, color=color, ecolor='black', capsize=3)
            xticks = hours[0:len(hours):3]
            ax.set_xticks(xticks)
            ax.set_xticklabels([f'{str(hour).zfill(2)}:00Z' for hour in xticks])

            ylim = param_ranges[param]
            if not ylim is None:
                (ymin, ymax) = ylim
                ax.set_ylim([ymin, ymax])

            plt.gca().xaxis.grid(True, linestyle='--')
            plt.gca().yaxis.grid(True, linestyle='--')
            plt.title(title)

            #if side_legend is None:
            #    plt.legend(loc='best')
            #else:
            #    plt.legend(bbox_to_anchor=(1,1), loc="upper left")

            # padding outfile name
            if outdir is not None:
                plt.savefig(f'{outdir}/{prefix}.png', bbox_inches='tight')
            plt.close()
print ("All done")
