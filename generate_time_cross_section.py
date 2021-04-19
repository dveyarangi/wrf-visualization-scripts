import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.pylab as pl
import datetime as dt
from datasets.wrf_dataset import WRFDataset
import datasets.surface_dataset as sid
from datasets.surface_israel_dataset import IsraelSurfaceDataset
import datasets.util as util
import numpy as np
from time_series import Series
from plot_profile import plot_time_series
import stations as st
from  profile_database import ProfileDatabase

from mpl_toolkits.basemap import Basemap
import os

from station import WeatherStation

os.environ["PROJ_LIB"] = "D:\Dev\Anaconda3\Library\share\proj"
domains = [ "d03"]
#dt.datetime(2013, 7, 12, 18, 00), dt.datetime(2013, 7, 14, 18, 00)
time_groups = [
            #(dt.datetime(2013, 7, 12, 18, 00), dt.datetime(2013, 7, 14, 18, 00)), \
            #(dt.datetime(2013, 8, 12, 18, 00),dt.datetime(2013, 8, 14, 18, 00)), \
            #(dt.datetime(2017, 11, 25, 18, 00),dt.datetime(2017, 11, 27, 18, 00)), \
            #(dt.datetime(2018, 2, 15, 18, 00),dt.datetime(2018, 2, 17, 18, 00)), \
            #(dt.datetime(2018, 4, 30, 18, 00),dt.datetime(2018, 5, 2, 18, 00)), \
            (dt.datetime(2020, 9, 13, 18, 00), dt.datetime(2020, 9, 16, 00, 00)), \
            (dt.datetime(2020, 9, 14, 18, 00), dt.datetime(2020, 9, 17, 00, 00)), \
            (dt.datetime(2020, 9, 15, 18, 00), dt.datetime(2020, 9, 18, 00, 00)) \
 \
    ]
base_wrf_dir = r"E:\meteo\urban-wrf\wrfout\\"

configs = ['bulk']
config = 'bulk'

db = ProfileDatabase()
all_datasets = {}
for config in configs:
    for domain in domains:
        ds_label = f"WRF {domain} {config}"
        dataset = WRFDataset(f"{base_wrf_dir}\\{config}", domain)
        all_datasets[ds_label] = dataset
        db.register_dataset(ds_label, dataset)

dataset_labels = [f"WRF {domain} {config}"]

datasets = []

minh = 0
maxh = 3500 # height of loaded data

max_h = 2500 # height of rendered data
h_step = 100
#max_h = 700 # height of rendered data
#h_step = 25
params = ["wvel_ms", "wdir_deg", "temp_k", "u_ms", "v_ms", "w_ms", "theta_k", "qvapor"]

for label in dataset_labels:
    datasets.append(db.get_dataset(label, minh, maxh, params))

# prepare arrays for statistics:
heights = db.get_heights(minh, maxh)

####################################################
# caching all relevant data:


tags = ['config1']

station_lat = 32.58
station_lon = 35.36

station = WeatherStation("test", station_lat, station_lon, 0)
baseOutDir = f"plots_2/config1/time_profile/{config}"
os.makedirs(baseOutDir, exist_ok=True)
all_profiles = {}


for ds in datasets:
    for (start_time, end_time) in time_groups:

        ftimes = ds.ds.get_datetimes(start_time, end_time)[6:55][::2]

        fhours = np.zeros((len(ftimes)))
        # np.arange(0, (end_time - start_time).total_seconds() / 3600 + 1, 1)
        ref_profile = ds.get_profile(start_time, 0, station)
        ground_hgt_msl = ref_profile.heights[0]
        heights = np.arange(ground_hgt_msl, ground_hgt_msl+max_h+h_step, h_step)
        heights_labels = np.zeros((len(heights)))
        plt.figure(figsize=(12, 6))


        ugrid = np.zeros( (len(heights), len(ftimes)) )
        vgrid = np.zeros( (len(heights), len(ftimes)) )
        wgrid = np.zeros((len(heights), len(ftimes)))
        thgrid = np.zeros((len(heights), len(ftimes)))
        tgrid = np.zeros((len(heights), len(ftimes)))
        qgrid = np.zeros((len(heights), len(ftimes)))

        profiles = {}

        for xidx, ftime in enumerate(ftimes):


            fhours[xidx] = (ftime - start_time).total_seconds() / 3600
            p = ds.get_profile(start_time, fhours[xidx], station)


            ip = p.interpolate(heights)
            fhours[xidx] -= 3

            profiles[fhours[xidx]] = p

            for yidx, hgt in enumerate(heights):
                wgrid[yidx, xidx] = ip.values["w_ms"][yidx]
                ugrid[yidx,xidx] = ip.values["u_ms"][yidx]
                vgrid[yidx,xidx] = ip.values["v_ms"][yidx]
                thgrid[yidx, xidx] = ip.values["theta_k"][yidx]
                tgrid[yidx, xidx] = ip.values["temp_k"][yidx]
                qgrid[yidx, xidx] = ip.values["qvapor"][yidx]

                heights_labels[yidx] = int(heights[yidx]-ground_hgt_msl)


        #########################################################################################

        wmag = (ugrid**2+vgrid**2)**0.5

        label_time = start_time + dt.timedelta(hours=6)
        title = f" Wind {domain} {label_time.strftime('%Y-%m-%d %H')}Z, " + \
                f" lat: {station_lat}, lon: {station_lon}"
        plt.title(title)

        x, y = np.meshgrid(fhours, heights_labels)

        #       [plt.plot([dot_x, dot_x], [0, dot_y], '-', linewidth=3) for dot_x, dot_y in zip(fhours, heights)]
        #       [plt.plot([0, dot_x], [dot_y, dot_y], '-', linewidth=3) for dot_x, dot_y in zip(fhours, heights)]
        cmap = plt.cm.rainbow
        #bounds = range(0, 16, 2)
        #norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        #plot = plt.contourf(x, y, wgrid, cmap=cmap,  vmin=-0.3, vmax=0.3)
        #plot = plt.contour(x, y, wgrid,  vmin=-0.3, vmax=0.3)

        wind_quiver = plt.quiver(x, y, ugrid, vgrid, color='grey', scale_units='xy', scale=5, linewidths=0.5)

        w_contours = plt.contour(x, y, wgrid, [0], linewidths=2, colors='green', linestyles='solid')
        plt.clabel(w_contours, inline=True, fmt='%.02f', fontsize=6)
        w_contours = plt.contour(x, y, wgrid, np.arange(0.05, 5, 0.05), linewidths=2, linestyles='solid',
                                      colors='red')
        plt.clabel(w_contours, inline=True, fmt='%.02f', fontsize=6)
        w_contours = plt.contour(x, y, wgrid, np.arange(-5, 0, 0.05), linewidths=2, linestyles='solid',
                                      colors='blue')
        plt.clabel(w_contours, inline=True, fmt='%.02f', fontsize=6)

        plt.xlabel("Hour [Summer Local Time]")
        plt.ylabel("Height [M]")
        plt.xticks(fhours)
        plt.yticks(heights_labels)
        #m = plt.cm.ScalarMappable(cmap=cmap)
        #m.set_array(wgrid)
        #m.set_clim(-0.3, 0.3)
        #plt.colorbar(m)
        plt.grid()
        filename = f"timecross_{label_time.strftime('%Y%m%d')}LST_{max_h}m_contours_wind_{station_lat}+{station_lon}.png"
        plt.tight_layout()

        plt.savefig(f'{baseOutDir}/{filename}')
        #plt.show()
        plt.clf()

        ##############################################################################
        title = f" Potential temp {domain} {label_time.strftime('%Y-%m-%d %H')}Z, " + \
                f" lat: {station_lat}, lon: {station_lon}"
        plt.title(title)

        x, y = np.meshgrid(fhours, heights_labels)

        #       [plt.plot([dot_x, dot_x], [0, dot_y], '-', linewidth=3) for dot_x, dot_y in zip(fhours, heights)]
        #       [plt.plot([0, dot_x], [dot_y, dot_y], '-', linewidth=3) for dot_x, dot_y in zip(fhours, heights)]
        cmap = plt.cm.rainbow
        #bounds = range(0, 16, 2)
        #norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        plot = plt.contourf(x, y, qgrid, cmap=cmap, levels=np.arange(0, 0.02, 0.001) )
        plt.colorbar(plot)
        plot = plt.contour(x, y, thgrid, colors='black', levels=range(0, 400, 1))

        plt.clabel(plot, inline=True, fmt='%.0f', fontsize=6)

        plt.xlabel("Hour [Summer Local Time]")
        plt.ylabel("Height [M]")
        plt.xticks(fhours)
        plt.yticks(heights_labels)
        #m = plt.cm.ScalarMappable(cmap=cmap)
        #m.set_array(thgrid)
        #m.set_clim(-0.3, 0.3)
        #plt.colorbar(m)
        plt.grid()
        filename = f"timecross_{label_time.strftime('%Y%m%d')}LST_{max_h}m_contours_potemp_qvapor_{station_lat}+{station_lon}.png"
        plt.tight_layout()

        plt.savefig(f'{baseOutDir}/{filename}')
        #plt.show()
        plt.clf()
        # cb = m.colorbar(wind_quiver, location='left')
        ###########################################################################################

        title = f" Wind {domain} {label_time.strftime('%Y-%m-%d %H')}Z, " + \
                f" lat: {station_lat}, lon: {station_lon}"
        plt.title(title)


        wmag = (ugrid ** 2 + vgrid ** 2) ** 0.5

        x, y = np.meshgrid(fhours, heights_labels)

        #       [plt.plot([dot_x, dot_x], [0, dot_y], '-', linewidth=3) for dot_x, dot_y in zip(fhours, heights)]
        #       [plt.plot([0, dot_x], [dot_y, dot_y], '-', linewidth=3) for dot_x, dot_y in zip(fhours, heights)]
        cmap = plt.cm.rainbow
        bounds = range(0, 16, 2)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        plot = plt.barbs(x, y, ugrid, vgrid, wmag, cmap=cmap, norm=norm, length=7, linewidth=2)
        plt.xlabel("Hour [Summer Local Time]")
        plt.ylabel("Height [M]")
        plt.xticks(fhours)
        plt.yticks(heights_labels)
        plt.colorbar(plot, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds)
        plt.grid()
        plt.tight_layout()

        filename = f"timecross_{label_time.strftime('%Y%m%d')}LST_{max_h}m_uv_barbs_{station_lat}+{station_lon}.png"
        plt.savefig(f'{baseOutDir}/{filename}')
        plt.clf()
