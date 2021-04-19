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


from mpl_toolkits.basemap import Basemap

domain = 'd04'

p1_lat = 32.68
p1_lon = 34.82
station_lat = 32.58
station_lon = 35.36

p2_lat = p1_lat + 1.5*(station_lat - p1_lat)
p2_lon = p1_lon + 1.5*(station_lon - p1_lon)

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

stations_data = {}

lons = xlon[0, :]
lats = xlat[:, 0]
lonmin = lons[0]
lonmax = lons[-1]
latmin = lats[0]
latmax = lats[-1]


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
stations = [st.stations[station_name] for station_name in station_names]

lon_min = 34.25
lon_max = 35.842
lat_min = 32.2
lat_max = 33

for station_name in station_names:
    station = st.stations[station_name]
    if station.lon >= lon_min and station.lon <= lon_max and station.lat >= lat_min and station.lat >= lat_max:
        print(station_name)
    stations.append(st.stations[station_name])


plt.figure(figsize=(8, 8))
# plt.tight_layout()
###########################
ax = plt.subplot(1, 1, 1)

dataset = WRFDataset(f"{base_wrf_dir}\\bulk", domain)
#title = f" {domain} {start_time.strftime('%Y-%m-%d %H')}Z+{int((curr_time - start_time).total_seconds() / 3600)}hrs, HGT"
#prefix = f'surface_map_{start_time.strftime("%Y%m%d%H")}_{int((curr_time - start_time).total_seconds() / 3600)}hrs_{domain}_{config}'

plt.title(f"Domain {domain}")
m = Basemap(projection='merc', llcrnrlat=latmin, urcrnrlat=latmax, \
            llcrnrlon=lonmin, urcrnrlon=lonmax, lat_ts=5, resolution='h')
# m.pcolormesh(xlon, xlat, hgt,latlon=True,cmap='terrain', vmin=-150, vmax=500)
# m.contourf(xlon, xlat, hgt,latlon=True, cmap='terrain', vmin=-150, vmax=500)


m.drawcoastlines()
m.fillcontinents(color='green', lake_color='aqua')
m.drawmapboundary()
m.drawstates()
m.drawcountries()
parallels = np.arange(0., 81, 0.2)
# labels = [left,right,top,bottom]
m.drawparallels(parallels, labels=[False, True, True, False])
meridians = np.arange(10., 351., 0.2)
m.drawmeridians(meridians, labels=[True, False, False, True])
lons, lats = m(station_lon, station_lat)
#m.scatter(lons, lats, marker='o', color='r', zorder=5)


cross_lats = [p1_lat, p2_lat]
cross_lons = [p1_lon, p2_lon]

st_lons = []
st_lats = []
#for station in stations:
#    (mx, my) = m(station.lon, station.lat)
#    st_lons.append(mx)
#    st_lats.append(my)

xlonflat = xlon.ravel()
xlatflat = xlat.ravel()
for idx in range(0, len(xlonflat)):
    (mx, my) = m(xlonflat[idx], xlatflat[idx])
    st_lons.append(mx)
    st_lats.append(my)

# plot points as red dots
#m.scatter(st_lons, st_lats, marker='.', color='black', zorder=5, s=1)


plt.title(f"{domain.upper()}, {xlat.shape[1]}x{xlat.shape[0]}, ~0.5km")
#points = [m(lon_min, lat_min),m(lon_min, lat_max),m(lon_max, lat_max),m(lon_max, lat_min)]
#poly = Polygon( points, facecolor='red', alpha=0.4 )
#plt.gca().add_patch(poly)

#m.plot(cross_lons, cross_lats, 'o-', markersize=5, linewidth=1, latlon=True)
plt.show()