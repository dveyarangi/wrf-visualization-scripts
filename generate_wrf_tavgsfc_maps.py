import matplotlib.pyplot as plt
import matplotlib.colors as colors
import xarray as xr

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


archive_dir = r'E:\meteo\urban-wrf'
types = ['no_tsk_tavg']

params = {'TAVGSFC': (275, 300, 26) }
#era_ds = xr.open_dataset(r'E:\meteo\urban-wrf\era5\2017\era5_20170926_an_sfc_0.grib', engine='cfgrib')
dates =  [ #'2013071218',
           '2017-11-25_18',
           '2018-02-15_18',
           #'2018043018',
           #'2020091318'
        ]

for date in dates:

    input_dir = f'{archive_dir}\\'


    for param in params:
        for domain in ['d03', 'd04']:
            plt.figure(figsize=(10, 10))

            title = f'{param} {domain} {date}'
            print(f"Plotting {title}...")
            ax = plt.subplot(1, 1, 1)
            filename = f'{input_dir}/met_em.{domain}.{date}%3A00%3A00.nc'
            print(filename)
            ds = util.load_dataset(filename)

            xlon = ds.variables["XLONG_M"][0]  # [::4,::4]
            xlat = ds.variables["XLAT_M"][0]  # [::4,::4]
            tsk = ds.variables[param][0]  # [::4,::4]
            hgt = ds.variables["HGT_M"][0]
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
            plt.savefig(f'plots/{date}_{param}_{domain}.png')
            plt.clf()