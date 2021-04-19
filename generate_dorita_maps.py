import matplotlib.pyplot as plt
import matplotlib.colors as colors

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
import wrf
#from netCDF4 import Dataset
import pytz
from wrf import (getvar, to_np, get_cartopy, latlon_coords, vertcross,
                 cartopy_xlim, cartopy_ylim, interpline, CoordPair)

local_tz = pytz.timezone('Etc/GMT-3')


p1_lat = 32.68
p1_lon = 34.82
station_lat = 32.58
station_lon = 35.36

p2_lat = p1_lat + 1.5*(station_lat - p1_lat)
p2_lon = p1_lon + 1.5*(station_lon - p1_lon)

cross_start = CoordPair(lat=p1_lat, lon=p1_lon)
cross_end   = CoordPair(lat=p2_lat, lon=p2_lon)

time_groups = [
           #(dt.datetime(2020, 9, 13, 18, 00), dt.datetime(2020, 9, 16, 00, 00)), \
           #(dt.datetime(2020, 9, 14, 18, 00), dt.datetime(2020, 9, 17, 00, 00)), \
           (dt.datetime(2020, 9, 15, 18, 00), dt.datetime(2020, 9, 18, 00, 00)) \
 \
    ]
lonmin = 34.25
lonmax = 35.842
latmin = 32.2
latmax = 33

min_height = -100
tag = 'config1'
domains = ["d03"]
datasets = {}
config = "bulk"
base_wrf_dir = r"E:\meteo\urban-wrf\wrfout\\"
for domain in domains:
     datasets [f"WRF {domain} {config}"] = WRFDataset(f"{base_wrf_dir}\\{config}", domain)

baseOutDir = f"plots_2/{tag}/maps"


def fill_cross_section(cross, thrsh):

    cross_filled = np.ma.copy(to_np(cross))

    # For each cross section column, find the first index with non-missing
    # values and copy these to the missing elements below.
    for i in range(cross_filled.shape[-1]):
        column_vals = cross_filled[:, i]

        first_idx = int(np.transpose((column_vals > thrsh).nonzero())[0])
        cross_filled[0:first_idx, i] = cross_filled[first_idx, i]

    return cross_filled

for time_range in time_groups:
    (start_time, end_time) = time_range
    label_time = start_time + dt.timedelta(hours=6)
    for domain in domains:
        dataset = datasets[f"WRF {domain} {config}"]

        outdir = f'{baseOutDir}/{config}/{label_time.strftime("%Y%m%d")}/'
        os.makedirs(outdir, exist_ok=True)
        ftimes = []
        curr_time = start_time
        while curr_time <= end_time:
            ftimes.append(curr_time)
            curr_time = curr_time + dt.timedelta(hours=0.5)

        for curr_time in ftimes:
            forecast_minute = (curr_time - start_time).total_seconds()/60
            if forecast_minute < 180: # skipping initialization time
                continue

            fhour_str = f'{((curr_time - start_time).total_seconds() / 3600):.1f}'


            path = dataset.create_filename(start_time, forecast_minute)
            #print(f'{curr_time} - {path}')
            ds = util.load_dataset(path)
            if ds is None:
                continue

            lu = ds.variables["LU_INDEX"][0]
            landmask = ds.variables["LANDMASK"][0]

            hgt = ds.variables["HGT"][0]
            (h, w) = hgt.shape
            # hgt = hgt[::4,::4]
            xlon = ds.variables["XLONG"][0]  # [::4,::4]
            xlat = ds.variables["XLAT"][0]  # [::4,::4]
            clon = ds.variables["XLONG"][0].copy()
            clat = ds.variables["XLAT"][0].copy()

            pblh = ds.variables["PBLH"][0]  # [::4,::4]
            ugrid = ds.variables["U10"][0]  # [::4,::4]
            vgrid = ds.variables["V10"][0]  # [::4,::4]
            t2mgrid = ds.variables["T2"][0] - 273.15

            psfcgrid = ds.variables["PSFC"][0]
            pTotgrid = (ds.variables["PB"][0] + ds.variables["P"][0])
            tgrid = (ds.variables["T"][0] + 300.0) * (
                        1. / 100000 * pTotgrid) ** (2. / 7)
            rhgrid = wrf.rh(ds.variables["Q2"], ds.variables["PSFC"], ds.variables["T2"])[0]
            rhgrid2 = wrf.rh(ds.variables["QVAPOR"][0], pTotgrid, tgrid)[0]

            ###########################
            plt.figure(figsize=(10, 6))

            disp_time = curr_time+dt.timedelta(hours=3)
            title = f"{domain} {disp_time.strftime('%Y-%m-%d %H:%M')} LST (+{fhour_str}hrs), PBLH HGT"
            print(f"Plotting {title}...")
            plt.title(title)
            m = Basemap(projection='merc', llcrnrlat=latmin, urcrnrlat=latmax, \
                                  llcrnrlon=lonmin,urcrnrlon=lonmax,lat_ts=5,resolution='h')

            pblh_cmap = 'rainbow'
            #t2m_mesh = m.pcolormesh(clon, clat, t2mgrid, latlon=True, vmin=10, vmax=30, cmap=t2m_cmap)
            pblh_mesh = m.contourf(xlon, xlat, pblh, latlon=True, cmap=pblh_cmap, levels=np.linspace(0, 2000, 21))
            cb = m.colorbar(pblh_mesh)

            m.drawmapboundary()
            m.drawstates()
            m.drawcountries()
            m.drawcoastlines(linewidth=2)
            topography = hgt.copy()
            topography[np.where(landmask == 0)] = np.NaN
            m.contour( xlon, xlat, topography, 10, colors='white', vmin=-150, vmax=2000, latlon=True, linewidths=0.5)
            m.plot([station_lon], [station_lat], 'x', color='red', markersize=10, linewidth=0.5, latlon=True)

            p = 3

            product_name = 'pblh_height'
            prefix = f'surface_map_{product_name}_{label_time.strftime("%Y%m%d")}_{fhour_str}hrs_{domain}_{config}'
            plt.tight_layout()
            plt.savefig(f'{outdir}/{prefix}.png')
            #plt.show()
            plt.close()

            plt.figure(figsize=(10, 6))
            disp_time = curr_time+dt.timedelta(hours=3)
            title = f"{domain} {disp_time.strftime('%Y-%m-%d %H:%M')} LST (+{fhour_str}hrs), T2 W10 HGT"

            print(f"Plotting {title}...")
            plt.title(title)
            m = Basemap(projection='merc', llcrnrlat=latmin, urcrnrlat=latmax, \
                                  llcrnrlon=lonmin,urcrnrlon=lonmax,lat_ts=5,resolution='h')

            t2m_cmap = 'jet'
            #t2m_mesh = m.pcolormesh(clon, clat, t2mgrid, latlon=True, vmin=10, vmax=30, cmap=t2m_cmap)
            t2m_mesh = m.contourf(xlon, xlat, t2mgrid, latlon=True, cmap=t2m_cmap, levels=np.linspace(18, 37, 20))
            cb = m.colorbar(t2m_mesh)

            m.drawmapboundary()
            m.drawstates()
            m.drawcountries()
            m.drawcoastlines(linewidth=2)
            topography = hgt.copy()
            topography[np.where(landmask == 0)] = np.NaN
            m.contour( xlon, xlat, topography, 10, colors='grey', vmin=-150, vmax=2000, latlon=True, linewidths=0.5)
            p = 3

            w_mag = np.sqrt(ugrid**2+vgrid**2)

            cdict = {'red':   [(0.0,  0.75, 0.35),
                               (1.0,  0.0, 0.0)],

                     'green': [(0.00, 0.0, 0.0),
                               (1.0,  0.0, 0.0),],

                     'blue':  [(0.0,  0.75, 0.35),
                               (1.0,  0.0, 0.0),]}
            red_black_colormap = colors.LinearSegmentedColormap('red_black', cdict)
            wind_quiver = m.quiver(xlon[::p,::p], xlat[::p,::p], ugrid[::p,::p], vgrid[::p,::p], w_mag[::p,::p], cmap=red_black_colormap, linewidths=1, latlon=True, scale_units='xy',scale=0.001)
            #cb = m.colorbar(wind_quiver, location='left')
            #cb.ax.yaxis.set_ticks_position('left')
            m.plot([station_lon], [station_lat], 'x', color='red', markersize=10, linewidth=0.5, latlon=True)
            plt.tight_layout()
            product_name = 'temp_wind_height'
            prefix = f'surface_map_{product_name}_{label_time.strftime("%Y%m%d")}_{fhour_str}hrs_{domain}_{config}'

            plt.savefig(f'{outdir}/{prefix}.png')
            #plt.show()
            plt.close()


            fig = plt.figure(figsize=(10, 6))
            # Get the WRF variables
            # Get the WRF variables
            ht = getvar(ds, "z", timeidx=-1)
            ter = getvar(ds, "ter", timeidx=-1)
            qvapor = getvar(ds, "QVAPOR", timeidx=-1)*1000 # convert to g/kg
            th = getvar(ds, "th", timeidx=-1)
            w = getvar(ds, "W", timeidx=-1)[1:,:,:]
            u = getvar(ds, "U", timeidx=-1)
            v = getvar(ds, "V", timeidx=-1)


            max_h = 11
            #max_dbz = getvar(ds, "mdbz", timeidx=-1)
            #Z = 10 ** (dbz / 10.)  # Use linear Z for interpolation

            # Compute the vertical cross-section interpolation.  Also, include the
            # lat/lon points along the cross-section in the metadata by setting latlon
            # to True.
            w_cross = vertcross(w, ht, wrfin=ds, start_point=cross_start, end_point=cross_end, latlon=True, meta=True)
            qvapor_cross = vertcross(qvapor, ht, wrfin=ds, start_point=cross_start, end_point=cross_end, latlon=True, meta=True)
            th_cross = vertcross(th, ht, wrfin=ds, start_point=cross_start, end_point=cross_end, latlon=True, meta=True)
            u_cross = vertcross(u, ht, wrfin=ds, start_point=cross_start, end_point=cross_end, latlon=True, meta=True)
            v_cross = vertcross(v, ht, wrfin=ds, start_point=cross_start, end_point=cross_end, latlon=True, meta=True)

            #rh_cross_filled = fill_cross_section(rh_cross, 0)
            #eth_cross_filled = fill_cross_section(rh_cross, -300)

            # Get the terrain heights along the cross section line
            ter_line = interpline(ter, wrfin=ds, start_point=cross_start,
                                  end_point=cross_end)

            ter_line[np.where(ter_line < min_height)] = min_height

            # Get the lat/lon points
            lats, lons = latlon_coords(w)

            # Get the cartopy projection object
            #cart_proj = get_cartopy(w)

            # Create the figure
            ######################################################################
            #

            fig = plt.figure(figsize=(10, 6))
            ax_cross = plt.axes()


            # Make the cross section plot for dbz
            xs = np.arange(0, qvapor_cross.shape[-1], 1)
            ys = to_np(qvapor_cross.coords["vertical"])[0:max_h]
            qvapor_contours = ax_cross.contourf(xs, ys, to_np(qvapor_cross)[0:max_h], levels=np.linspace(0, 20, 11), cmap='CMRmap_r')
            # Add the color bar
            cb_qvapor = fig.colorbar(qvapor_contours, ax=ax_cross)
            cb_qvapor.ax.tick_params(labelsize=8)

            th_levels = levels = range(0,450)
            th_contours = ax_cross.contour(xs, ys, to_np(th_cross)[0:max_h], th_levels, linewidths=0.5, colors='black')
            ax_cross.clabel(th_contours, inline=True, fmt='%.0f', fontsize=6)

            wind_quiver = ax_cross.quiver(xs, ys, u_cross[0:max_h], v_cross[0:max_h], cmap=red_black_colormap, linewidths=1)

            # Fill in the mountain area
            ht_fill = ax_cross.fill_between(xs, min_height, to_np(ter_line),
                                            facecolor="darkgrey")

            # Set the x-ticks to use latitude and longitude labels

            coord_pairs = to_np(qvapor_cross.coords["xy_loc"])
            x_ticks = np.arange(coord_pairs.shape[0])
            x_labels = [pair.latlon_str(fmt="{:.2f}\n{:.2f}") for pair in to_np(coord_pairs)]

            # Set the desired number of x ticks below
            num_ticks = 9
            thin = int((len(x_ticks) / num_ticks) + .5)
            xticks = x_ticks[::thin]
            ax_cross.set_xticks(x_ticks[::thin])
            ax_cross.set_xticklabels(x_labels[::thin], fontsize=8)

            # Set the x-axis and  y-axis labels
            ax_cross.set_xlabel("Latitude, Longitude", fontsize=12)
            ax_cross.set_ylabel("Height (m)", fontsize=12)

            ax_cross.axvline(x=len(x_ticks)/1.5, linestyle=':', color='red', linewidth=1)
            # Add a title
            #ax_cross.set_title("Cross-Section of Reflectivity (dBZ)", {"fontsize": 14})
            title = f"{domain} {disp_time.strftime('%Y-%m-%d %H:%M')} LST (+{fhour_str}hrs), UV SH TH"
            product_name = 'uv_sh_th'
            prefix = f'space_cross_{product_name}_{label_time.strftime("%Y%m%d")}_{fhour_str}hrs_{domain}_{config}'
            print(f"Plotting {title}...")
            ax_cross.set_title(title)


            plt.tight_layout()
            plt.savefig(f'{outdir}/{prefix}.png')
            #plt.show()
            plt.close()


            ######################################################################
            #
            max_h = 11
            # Create the figure
            plt.figure(figsize=(8, 6))
            ax_cross = plt.axes()

            ys = to_np(qvapor_cross.coords["vertical"])[0:max_h]
            # Fill in the mountain area
            ht_fill = ax_cross.fill_between(xs, min_height, to_np(ter_line),
                                            facecolor="darkgrey")

            wind_quiver = ax_cross.quiver(xs, ys, u_cross[0:max_h], v_cross[0:max_h], cmap=red_black_colormap, scale_units='xy', scale=5, linewidths=0.5)

            w_contours = ax_cross.contour(xs, ys, to_np(w_cross)[0:max_h], [0], linewidths=1, colors='green')
            ax_cross.clabel(w_contours, inline=True, fmt='%.02f', fontsize=6)
            w_contours = ax_cross.contour(xs, ys, to_np(w_cross)[0:max_h], np.arange(0.25, 5, 0.25), linewidths=1, colors='red')
            ax_cross.clabel(w_contours, inline=True, fmt='%.02f', fontsize=6)
            w_contours = ax_cross.contour(xs, ys, to_np(w_cross)[0:max_h], np.arange(-5, -0.25, 0.25), linewidths=1, colors='blue')
            ax_cross.clabel(w_contours, inline=True, fmt='%.02f', fontsize=6)


            ax_cross.set_xticks(x_ticks[::thin])
            ax_cross.set_xticklabels(x_labels[::thin], fontsize=8)

            # Set the x-axis and  y-axis labels
            ax_cross.set_xlabel("Latitude, Longitude", fontsize=12)
            ax_cross.set_ylabel("Height (m)", fontsize=12)
            ax_cross.axvline(x=len(x_ticks)/1.5, linestyle=':', color='red', linewidth=1)

            title = f"{domain} {disp_time.strftime('%Y-%m-%d %H:%M')} LST (+{fhour_str}hrs), W UV"
            product_name = 'w_uv'
            prefix = f'space_cross_{product_name}_{label_time.strftime("%Y%m%d")}_{fhour_str}hrs_{domain}_{config}'
            print(f"Plotting {title}...")
            ax_cross.set_title(title)
            plt.tight_layout()
            plt.savefig(f'{outdir}/{prefix}.png')
            #plt.show()
