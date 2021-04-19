import matplotlib.pyplot as plt
import matplotlib.colors as colors
import xarray as xr
import math
import matplotlib.pylab as pl
import datetime as dt
from datasets.wrf_dataset import WRFDataset
#import datasets.surface_dataset as sid
#from datasets.surface_israel_dataset import IsraelSurfaceDataset
import datasets.util as util
import numpy as np
#import stations as st
#from matplotlib.patches import Polygon

from mpl_toolkits.basemap import Basemap
import os


archive_dir = r'E:\meteo\urban-wrf\wrfinput'
#types = ['no_tsk_tavg']
types = ['bulk', 'bulk_sst', 'bulk_no_tsk']
#types = ['mod_metgrid']

params = {'SST': (291, 297, 25), 'TSK': (275, 304, 26) }
params = {'TSK': (275, 304, 26) }
#era_ds = xr.open_dataset(r'E:\meteo\urban-wrf\era5\2017\era5_20170926_an_sfc_0.grib', engine='cfgrib')
dates =  [ #'2013071218',
           '2017112518',
           #'2018021518',
           #'2018043018',
           #'2020091318'
        ]

for date in dates:

    input_dir = f'{archive_dir}\\{date}'
    '''
    for param in params:
        for domain in ['d03']:
            for idx, type in enumerate(types):
                title = f'{param} {type} {domain}'
                ax = plt.subplot(1, 3, idx + 1)
                filename = f'{input_dir}/{type}/wrfinput_{domain}'
                ds = util.load_dataset(filename)

                param_arr = ds.variables[param][0]  # [::4,::4]

                if param == 'SST':

                    max = int(param_arr.max())+2
                    min = max-6
                    steps = 3*(max - min + 1)
                else:
                    min = int(param_arr.min())-2
                    max = int(param_arr.max())+2
                    steps = (max - min + 1)
                params[param] = (min, max, steps)'''

    for param in params:
        for domain in ['d04', 'd03']:
            plot_w = min(len(types), 3)
            plot_h = max(math.ceil(len(types)/3), 1)
            plt.figure(figsize=(7*plot_w,7*plot_h))
            plot_filename = f'plots/{date}_{param}_{domain}.png'
            print(f"Plotting {plot_filename}")
            for idx, type in enumerate(types):
                title = f'{param} {type} {domain} {date}'
                print(f" > Plotting {title}...")
                ax = plt.subplot(plot_h, plot_w, idx+1)
                filename = f'{input_dir}/{type}/wrfinput_{domain}'
                ds = util.load_dataset(filename)

                xlon = ds.variables["XLONG"][0]  # [::4,::4]
                xlat = ds.variables["XLAT"][0]  # [::4,::4]
                tsk = ds.variables[param][0]  # [::4,::4]
                hgt = ds.variables["HGT"][0]
                landmask = ds.variables["LANDMASK"][0]

                (h, w) = hgt.shape
                lons = xlon[0, :]
                lats = xlat[:, 0]
                lonmin = lons[0]
                lonmax = lons[-1]
                latmin = lats[0]
                latmax = lats[-1]

                m = Basemap(projection='merc', llcrnrlat=latmin, urcrnrlat=latmax, \
                            llcrnrlon=lonmin, urcrnrlon=lonmax, lat_ts=5, resolution='h')

                pblh_cmap = 'rainbow'
                # t2m_mesh = m.pcolormesh(clon, clat, t2mgrid, latlon=True, vmin=10, vmax=30, cmap=t2m_cmap)

                (vmin, vmax, steps) = params[param]
                #steps = int(vmax-vmin)+1
                pblh_mesh = m.contourf(xlon, xlat, tsk, latlon=True, cmap=pblh_cmap, levels=np.linspace(vmin, vmax, steps))
                cb = m.colorbar(pblh_mesh)
                pblh_cont = m.contour(xlon, xlat, tsk, latlon=True, colors='black',levels=np.linspace(vmin, vmax, steps), linewidths=0.5)

                m.drawmapboundary()
                m.drawstates()
                m.drawcountries()
                m.drawcoastlines(linewidth=2)
                topography = hgt.copy()
                topography[np.where(landmask == 0)] = np.NaN
                m.contour(xlon, xlat, topography, 10, colors='white', vmin=-150, vmax=2000, latlon=True, linewidths=0.5)

                plt.title(title)

            plt.tight_layout()
            #plt.show()
            plt.savefig(plot_filename)
            plt.clf()