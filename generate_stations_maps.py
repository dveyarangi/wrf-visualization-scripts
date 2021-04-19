import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.pylab as pl

import datetime as dt
from datasets.wrf_dataset import WRFDataset
import datasets.surface_dataset as sid
from datasets.surface_israel_dataset import IsraelSurfaceDataset
import datasets.util as util
import numpy as np
import stations as st
from matplotlib.patches import Polygon
from spatial_index import SpatialIndex

from mpl_toolkits.basemap import Basemap

domain = 'd03'

base_wrf_dir = r"E:\meteo\urban-wrf\wrfout\\"


time = dt.datetime(2020, 9, 13, 18, 00)
forecast_minute = 0
dataset = WRFDataset(f"{base_wrf_dir}\\bulk", domain)

path = dataset.create_filename(time, forecast_minute)
# print(f'{curr_time} - {path}')
ds = util.load_dataset(path)

lu = ds.variables["LU_INDEX"][0]
landmask = ds.variables["LANDMASK"][0]

hgt = ds.variables["HGT"][0]
(h, w) = hgt.shape
# hgt = hgt[::4,::4]
xlon = ds.variables["XLONG"][0]  # [::4,::4]
xlat = ds.variables["XLAT"][0]  # [::4,::4]
ugrid = ds.variables["U10"][0]  # [::4,::4]
vgrid = ds.variables["V10"][0]  # [::4,::4]


mlons = xlon[0, :]
mlats = xlat[:, 0]
lonmin = mlons[0]
lonmax = mlons[-1]
latmin = mlats[0]
latmax = mlats[-1]

#indexer = SpatialIndex()
#for (( i, j), val) in np.ndenumerate(xlon):
#    indexer.add(xlat[i, j], xlon[i, j], i, j)
station_names = [ \
                 'Afek', \
                 'Ein Karmel', \
                 'Haifa Refineries', \
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

all_stations = st.load_surface_stations(st.IMS_STATIONS_FILE)
#stations = [st.stations[station_name] for station_name in station_names]
wrfds = WRFDataset(f"{base_wrf_dir}\\bulk", domain)
(latmin,lonmin,latmax,lonmax) = wrfds.get_domain_coverage()
#(latmin,lonmin,latmax,lonmax) = st.get_aoi(all_stations)
stations = st.filter_by_aoi(all_stations, latmin,lonmin,latmax,lonmax)


plt.figure(figsize=(20, 20))
# plt.tight_layout()
###########################
ax = plt.subplot(1, 1, 1)

#title = f" {domain} {start_time.strftime('%Y-%m-%d %H')}Z+{int((curr_time - start_time).total_seconds() / 3600)}hrs, HGT"
#prefix = f'surface_map_{start_time.strftime("%Y%m%d%H")}_{int((curr_time - start_time).total_seconds() / 3600)}hrs_{domain}_{config}'

plt.title(f"Domain {domain}")
m = Basemap(projection='merc', llcrnrlat=latmin, urcrnrlat=latmax, \
            llcrnrlon=lonmin, urcrnrlon=lonmax, lat_ts=5, resolution='h')
# m.pcolormesh(xlon, xlat, hgt,latlon=True,cmap='terrain', vmin=-150, vmax=500)
#m.contourf(xlon, xlat, hgt,latlon=True, cmap='terrain', vmin=-150, vmax=500)

bathymetry = hgt.copy()
bathymetry[np.where(landmask > 0)] = np.NaN
topography = hgt.copy()
#topography[np.where(landmask == 0)] = np.NaN
hgt_mesh = m.contourf(xlon, xlat, topography, 35, latlon=True, cmap='terrain', vmin=-300, vmax=2600)
cb = m.colorbar(hgt_mesh)
m.contourf(xlon, xlat, bathymetry, 10, cmap='terrain', vmin=-150, vmax=2000, latlon=True)

m.drawcoastlines()
#m.fillcontinents(color='green', lake_color='aqua')
m.drawmapboundary()
m.drawstates()
m.drawcountries()
parallels = np.arange(0., 81, 0.5)
# labels = [left,right,top,bottom]
m.drawparallels(parallels, labels=[True, False, False, False])
meridians = np.arange(10., 351., 0.5)
m.drawmeridians(meridians, labels=[False, False, False, True])
#m.scatter(lons, lats, marker='o', color='r', zorder=5)

#grid_lons = []
#grid_lats = []

#xlonflat = xlon.ravel()
#xlatflat = xlat.ravel()
#for idx in range(0, len(xlonflat)):
#    (mx, my) = m(xlonflat[idx], xlatflat[idx])
#    grid_lons.append(mx)
#    grid_lats.append(my)

# plot points as red dots
#m.scatter(st_lons, st_lats, marker='.', color='black', zorder=5, s=1)
station_lats = np.zeros((len(stations)))
station_lons = np.zeros((len(stations)))

for idx, station in enumerate(stations):
    station_lats[idx] = station.lat
    station_lons[idx] = station.lon
(station_lons, station_lats) = m(station_lons, station_lats)
m.scatter(station_lons, station_lats, marker='o', color='black', zorder=5, s=1)
for i, station in enumerate(stations):
    #(pi, pj, plat, plon) = indexer.get_closest_index(station.lat, station.lon)
    #model_hgt = hgt[pi,pj]
    #tation_hgt = station.hgt
    anno = f"{station.wmoid}"
    '''   print(f"{station.wmoid}, {station_hgt}, {model_hgt:.1f}, "
          f"{hgt[pi+1,pj]:.1f}, "
          f"{hgt[pi-1,pj]:.1f}, "
          f"{hgt[pi,pj-1]:.1f}, "
          f"{hgt[pi,pj+1]:.1f}, "
          f"{station.lat:0.4f}, {station.lon:0.4f},"
          f"{plat:0.4f}, {plon:0.4f},"
          f"{xlat[pi+1,pj]:0.4f}, {xlon[pi+1,pj]:0.4f},"
          f"{xlat[pi-1,pj]:0.4f}, {xlon[pi-1,pj]:0.4f}, "
          f"{xlat[pi,pj-1]:0.4f}, {xlon[pi,pj-1]:0.4f},"
          f"{xlat[pi,pj+1]:0.4f}, {xlon[pi,pj+1]:0.4f},");
    '''
#    print(f"{xlat[pi-1,pj]} {xlon[pi-1,pj]}") # S
#    print(f"{xlat[pi+1,pj]} {xlon[pi+1,pj]}") # N
#    print(f"{xlat[pi,pj-1]} {xlon[pi,pj-1]}") # W
#    print(f"{xlat[pi,pj+1]} {xlon[pi,pj+1]}") # E
    ax.annotate(anno, (station_lons[i], station_lats[i]))

plt.savefig(f'plots/stations_{domain}.png')
plt.show()

