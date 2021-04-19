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
import wrf

from mpl_toolkits.basemap import Basemap
import os


LANDUSE_LABELS = [('EG Needle Forest', '#006600'),
                  ('EG Broad Forest' , '#007733'),
                  ('DD Needle Forest', '#005500'),
                  ('DD Broad Forest' , '#006633'),
                  ('Mixed Forests'   , '#009900'),
                  ('Closed Shrubs'   , '#669900'),
                  ('Open Shrubs'     , 'darkgoldenrod'),
                  ('Woody Savannas'  , '#99cc00'),
                  ('Savannas'        , '#ff9900'),
                  ('Grasslands'      , '#bbff00'),
                  ('Perm wetlands'   , '#009999'),
                  ('Croplands'       , '#ccff66'),
                  ('Urban/Built-Up'  , '#666699'),
                  ('Veg mosaic'      , '#00ff00'),
                  ('Snow and Ice'    , '#ccffff'),
                  ('Barren/Sparse Veg', '#999966'),
                  ('Water'           , 'steelblue'),
                  ('Wooded Tundra'   , '#00cc66'),
                  ('Mixed Tundra'    , '#00ff99'),
                  ('Barren Tundra'   , '#99ff99')]

def truncate_landuse(landuse):

    truncated_landuse = np.zeros(landuse.shape)
    unique, counts = np.unique(landuse, return_counts=True)

    cmap = plt.get_cmap('tab20')
    color_list = []
    labels = []
    for idx, value in enumerate(unique):
        (label, color) = LANDUSE_LABELS[round(value)-1]
        color_list.append(color)

        truncated_landuse[np.where(landuse == value)] = idx+1
        #print(round(value))
        labels.append(label)

    cmap = colors.ListedColormap(color_list);
    return (truncated_landuse, cmap, labels)


def rh(Q_kg, P_hpa, T_k):

    Tn = 240.7263
    A = 6.116441
    B = 621.9907
    m = 7.591386
    Q_g = 1000.*Q_kg
    T_c = T_k - 273.15

    Pw = Q_g*P_hpa / (Q_g+B)
    Pws = A*10**(m*T_c/(T_c+Tn))

    rh = 100. * Pw / Pws

    return rh


os.environ["PROJ_LIB"] = "D:\Dev\Anaconda3\Library\share\proj"

station_names = [ \
                 'Afek', \
                 'Ein Karmel', \
                 #'Haifa Refineries', \
                 'Haifa Technion', \
                 'Haifa University', \
                 'Shavei Zion', \
                 'Nesher',
                 'K.HAIM-REGAVIM',
                 'K.Hasidim',
                 'K.Bnyamin',
                 'K.Tivon',
                 'K.Yam',
                 'Shprinzak',
                 'N.Shanan',
                 'Ahuza',
                 'K.Bialik-Ofarim',
                 'IGUD',
                 'K.Ata'
                ]

stations = [st.stations[station_name] for station_name in station_names]

#
domains = [ "d04"]
#dt.datetime(2013, 7, 12, 18, 00), dt.datetime(2013, 7, 14, 18, 00)
time_groups = [
            #(dt.datetime(2013, 7, 12, 18, 00), dt.datetime(2013, 7, 14, 18, 00)), \
            #(dt.datetime(2013, 8, 12, 18, 00),dt.datetime(2013, 8, 14, 18, 00)), \
            (dt.datetime(2017, 9, 26, 18, 00),dt.datetime(2017, 9, 28, 18, 00)), \
            (dt.datetime(2017, 11, 25, 18, 00),dt.datetime(2017, 11, 27, 18, 00)), \
            (dt.datetime(2018, 2, 15, 18, 00),dt.datetime(2018, 2, 17, 18, 00)), \
            (dt.datetime(2018, 4, 30, 18, 00),dt.datetime(2018, 5, 2, 18, 00)), \
            (dt.datetime(2020, 9, 13, 18, 00), dt.datetime(2020, 9, 16, 00, 00)), \
            (dt.datetime(2020, 9, 14, 18, 00), dt.datetime(2020, 9, 17, 00, 00)), \
            (dt.datetime(2020, 9, 15, 18, 00), dt.datetime(2020, 9, 18, 00, 00)) \
 \
    ]


domain_pruning = { 'd04': (3, 5)}

tags = ['config1']
configs = ['bulk', 'bulk_sst'] #
#configs = ['bulk_sst'] #
#configs = ['bulk_d01_input_only'] # 'bulk_nofeedback'

surface_ds = sid.SurfaceDataset(sid.surface_archive_dir)

tag_datasets = {}

base_wrf_dir = r"E:\meteo\urban-wrf\wrfout\\"

for tag in tags:
    domain_datasets = {}
    for domain in domains:
        for config in configs:
            domain_datasets [f"WRF {domain} {config}"] = WRFDataset(f"{base_wrf_dir}\\{config}", domain)
    tag_datasets[tag] = domain_datasets

all_datasets = tag_datasets[tag]
baseOutDir = f"plots_tmp/{tag}/maps"


##########################################
# indexing plotting iterations for progress print:
total_times = 0
for time_range in time_groups:
    (start_time, end_time) = time_range
    total_times += int((end_time - start_time).total_seconds())/3600

total_plots = len(configs) * len(domains) * int(total_times)
plot_idx = 0

##########################################
# extracting map converage based on stations

lonmin = None
latmin = None
lonmax = None
latmax = None
for idx, station in enumerate(stations):
    if lonmin is None or lonmin > station.lon:
        lonmin = station.lon
    if lonmax is None or lonmax < station.lon:
        lonmax = station.lon
    if latmin is None or latmin > station.lat:
        latmin = station.lat
    if latmax is None or latmax < station.lat:
        latmax = station.lat

lonmin = lonmin - 0.1
lonmax = lonmax + 0.1
latmin = latmin - 0.05
latmax = latmax - 0.05

m = Basemap(projection='merc', llcrnrlat=latmin, urcrnrlat=latmax, \
            llcrnrlon=lonmin, urcrnrlon=lonmax, lat_ts=5, resolution='h')

##########################################

##########################################
for domain in domains:


    for time_range in time_groups:
        (start_time, end_time) = time_range

        ftimes = []
        curr_time = start_time
        while curr_time <= end_time:
            ftimes.append(curr_time)
            curr_time = curr_time + dt.timedelta(hours=1)

        ##########################################
        # calculating value ranges for plots
        ranges = {'w10mag': (10000, -10000), 't2m': (10000, -10000), 'rh2m': (10000, -10000)}

        for config in configs:
            dataset = all_datasets[f"WRF {domain} {config}"]

            for curr_time in ftimes:
                forecast_minute = (curr_time - start_time).total_seconds()/60
                #if forecast_minute < 360: # skipping initialization time
                #    continue
                path = dataset.create_filename(start_time, forecast_minute)
                #print(f'{curr_time} - {path}')
                ds = util.load_dataset(path)
                if ds is None:
                    continue

                ugrid = ds.variables["U10"][0]
                vgrid = ds.variables["V10"][0]
                t2mgrid = ds.variables["T2"][0] - 273.15
                rh2grid = wrf.rh(ds.variables["Q2"], ds.variables["PSFC"], ds.variables["T2"])[0]

                (vmin, vmax) = ranges['w10mag']
                wgrid = np.sqrt(ugrid**2 + vgrid**2)
                w10min = np.amin(wgrid)
                if w10min < vmin: vmin = w10min
                w10max = np.amax(wgrid)
                if w10max > vmax: vmax = w10max
                ranges['w10mag'] = (vmin, vmax)

                (vmin, vmax) = ranges['t2m']
                t2min = np.amin(t2mgrid)
                t2max = np.amax(t2mgrid)
                if t2min < vmin: vmin = t2min
                if t2max > vmax: vmax = t2max
                ranges['t2m'] = (vmin, vmax)

                (vmin, vmax) = ranges['rh2m']
                rh2min = np.amin(rh2grid)
                rh2max = np.amax(rh2grid)
                if rh2min < vmin: vmin = rh2min
                if rh2max > vmax: vmax = rh2max
                ranges['rh2m'] = (vmin, vmax)

        ##########################################
        # rendering maps
        for config in configs:
            dataset = all_datasets[f"WRF {domain} {config}"]

            outdir = f'{baseOutDir}/{config}/{start_time.strftime("%Y%m%d%H")}/{domain}'
            os.makedirs(outdir, exist_ok=True)

            for curr_time in ftimes:
                plot_idx += 1
                forecast_minute = (curr_time - start_time).total_seconds()/60
                #if forecast_minute < 360: # skipping initialization time
                #    continue
                path = dataset.create_filename(start_time, forecast_minute)
                #print(f'{curr_time} - {path}')
                ds = util.load_dataset(path)
                if ds is None:
                    continue

                lu = ds.variables["LU_INDEX"][0]
                landmask = ds.variables["LANDMASK"][0]

                hgt = ds.variables["HGT"][0]
                (h, w) = hgt.shape
                xlon = ds.variables["XLONG"][0]
                xlat = ds.variables["XLAT"][0]
                clon = xlon.copy()
                clat = xlat.copy()

                ugrid = ds.variables["U10"][0]
                vgrid = ds.variables["V10"][0]
                t2mgrid = ds.variables["T2"][0] - 273.15
                rh2grid = wrf.rh(ds.variables["Q2"], ds.variables["PSFC"], ds.variables["T2"])[0]

                #psfcgrid = ds.variables["PSFC"][0]
                #pTotgrid = (ds.variables["PB"][0] + ds.variables["P"][0])
                #tgrid = (ds.variables["T"][0] + 300.0) * (
                #            1. / 100000 * pTotgrid) ** (2. / 7)
                #rhgrid2 = wrf.rh(ds.variables["QVAPOR"][0], pTotgrid, tgrid)[0]
                #rhgrid = rh(ds.variables["QVAPOR"][0], pTotgrid/100., tgrid)[0]

                #for x in range(0,w-1):
                #    for y in range(0, h-1):
                #        dlat = (xlat[y+1,x]-xlat[y,x]+xlat[y+1,x+1]-xlat[y,x+1]) / 4
                #        dlon = (xlon[y, x+1]-xlon[y, x] + xlon[y+1, x+1]-xlon[y+1, x]) / 4
                #        clat[y,x] -= dlat
                #        clon[y,x] -= dlon

                station_lats = np.zeros((len(stations)))
                station_lons = np.zeros((len(stations)))
                station_us = np.zeros((len(stations)))
                station_vs = np.zeros((len(stations)))
                station_t2m = np.zeros((len(stations)))
                station_rh = np.zeros((len(stations)))
                model_us = np.zeros((len(stations)))
                model_vs = np.zeros((len(stations)))
                stations_data = {}

                lonmin = None
                lonmax = None
                latmin = None
                latmax = None
                for idx,station in enumerate(stations):
                    station_data = surface_ds.get_station_series( station, curr_time, params=['u10_ms', 'v10_ms', "temp2m_c", 'rh'])
                    station_lats[idx] = station.lat
                    station_lons[idx] = station.lon
                    if "u10_ms" in station_data:
                        station_us[idx] = station_data["u10_ms"]
                    else:
                        station_us[idx] = np.NaN
                    if "v10_ms" in station_data:
                        station_vs[idx] = station_data["v10_ms"]
                    else:
                        station_us[idx] = np.NaN
                    if "temp2m_c" in station_data:
                        station_t2m[idx] = station_data["temp2m_c"]
                    else:
                        station_t2m[idx] = np.NaN
                    if "rh" in station_data:
                        station_rh[idx] = station_data["rh"]
                    else:
                        station_rh[idx] = np.NaN


                    if len(station_data) > 0:
                        (pi, pj, plat, plon) = dataset.indexer.get_closest_index(station.lat, station.lon)
                        model_us[idx] = ugrid[pi,pj]
                        model_vs[idx] = vgrid[pi, pj]

                    stations_data[station] = station_data

                ###########################
                fig = plt.figure(figsize=(12, 6))

                ax = plt.subplot(1,2,1)
                title = f"WRF HGT"
                product_name = 'wind_height'
                prefix = f'surface_map_{product_name}_{start_time.strftime("%Y%m%d%H")}_{int((curr_time-start_time).total_seconds() / 3600)}hrs_{domain}_{config}'
                print(f" * {title}...")
                plt.title(title)
#                m = Basemap(projection='merc', llcrnrlat=latmin, urcrnrlat=latmax, \
#                                      llcrnrlon=lonmin,urcrnrlon=lonmax,lat_ts=5,resolution='h')
                bathymetry = hgt.copy()
                bathymetry[np.where(landmask > 0)] = np.NaN
                topography = hgt.copy()
                topography[np.where(landmask == 0)] = np.NaN
                hgt_mesh = m.contourf(xlon, xlat, topography, 15, latlon=True, cmap='terrain', vmin=-300, vmax=500)
                cb = m.colorbar(hgt_mesh)
                m.contourf( xlon, xlat,bathymetry, 10, cmap='terrain', vmin=-150, vmax=2000, latlon=True)
                (p1, p2) = domain_pruning[domain]

                w_mag = np.sqrt(ugrid**2+vgrid**2)

                cdict = {'red':   [(0.0,  0.0, 0.0),
                                   (1.0,  0.75, 0.35)],

                         'green': [(0.0,  0.0, 0.0),
                                   (1.00, 0.35, 0.35)],

                         'blue':  [(0.0,  0.0, 0.0),
                                   (1.0,  0.0, 0.0)]}
                #red_black_colormap = colors.LinearSegmentedColormap('red_black', cdict)
                (vmin,vmax) = ranges['w10mag']
                #norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
                #wind_quiver = m.quiver(xlon[::p1,::p1], xlat[::p1,::p1], ugrid[::p1,::p1], vgrid[::p1,::p1], w_mag[::p1,::p1], cmap=red_black_colormap, norm=norm,linewidth=1.5, latlon=True, scale_units='xy',scale=0.003)
                #cb = m.colorbar(wind_quiver, location='left')
                #cb.ax.yaxis.set_ticks_position('left')
                #m.quiver(station_lons, station_lats, model_us, model_vs, scale_units='xy',scale=0.001, latlon=True, linewidth=2)
                #m.quiver(station_lons, station_lats, station_us, station_vs, scale_units='xy',scale=0.001, latlon=True, linewidth=1, color='red')
                #plt.tight_layout()
                #plt.savefig(f'{outdir}/{prefix}')
                #plt.clf()
                ###################################################
                #fig, axes = plt.subplots(nrows=2, ncols=2)

                ax = plt.subplot(1,2, 2)
                title = f"WRF LANDUSE"
                print(f" * {title}...")

                plt.title(title)
#                m = Basemap(projection='merc', llcrnrlat=latmin, urcrnrlat=latmax, \
#                                      llcrnrlon=lonmin,urcrnrlon=lonmax,lat_ts=5,resolution='h')
                landuse, cmap, labels = truncate_landuse(lu)
                landuse_mesh = m.pcolormesh(clon, clat, landuse,latlon=True,cmap=cmap, vmin=1, vmax=len(labels))

                mappable = matplotlib.cm.ScalarMappable(cmap=cmap)
                mappable.set_array([])
                mappable.set_clim(-0.5, len(labels) + 0.5)
                cb = m.colorbar(mappable, ticks=np.linspace(0, len(labels), len(labels)))
                cb.ax.set_yticklabels(labels)
                cb.ax.tick_params(labelsize=8)

                plt.show()
                plt.close('all')