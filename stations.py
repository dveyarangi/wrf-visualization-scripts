import os
import datasets.archive_config as archive_config
from datasets.wyoming_sonde_dataset import WyomingSondeDataset
from station import WeatherStation
import pandas as pd

'''
stations = {
        'Haifa Technion': WeatherStation('Haifa Technion', 32.7736, 35.0223, 245),
        'Haifa University': WeatherStation('Haifa University', 32.7611, 35.0208, 475),
        'Haifa Refineries': WeatherStation('Haifa Refineries', 32.8034, 35.0548, 5),
        'Afek': WeatherStation('Afek', 32.8466, 35.1123, 10),
        'Ein Karmel': WeatherStation('Ein Karmel', 32.6808, 34.9594, 25),
        'Shavei Zion': WeatherStation('Shavei Zion', 32.9836, 35.0923, 25),
        'Nesher': WeatherStation('Nesher', 32.76994, 35.04226, 90),
        'K.HAIM-REGAVIM': WeatherStation('K.HAIM-REGAVIM', 32.830438, 35.056901, 7),
        'K.Hasidim': WeatherStation('K.Hasidim', 32.74263, 35.09444, 95),
        'K.Bnyamin': WeatherStation('K.Bnyamin', 32.78866, 35.08511, 5),
        'K.Tivon': WeatherStation('K.Tivon', 32.72180, 35.12940, 201),
        'K.Yam': WeatherStation('K.Yam', 32.85182, 35.07873, 27),
        'Shprinzak': WeatherStation('Shprinzak', 32.8226, 34.9651, 107),
        'N.Shanan': WeatherStation('N.Shanan', 32.78685, 35.02026, 240),
        'Ahuza': WeatherStation('Ahuza', 32.785247, 34.984667, 280),

        'K.Bialik-Ofarim': WeatherStation('K.Bialik-Ofarim', 32.8142, 35.07806, 4),
        'IGUD': WeatherStation('IGUD', 32.7895, 35.04066, 18),
        'K.Ata': WeatherStation('K.Ata', 32.8115, 35.11279, 65),

}
'''
IMS_STATIONS_FILE = 'etc/ims_stations.csv'


def load_surface_stations(filename):
    df = pd.read_csv(filename)
    stations = []
    for (idx,row) in df.iterrows():
        wmoid = row['Id'].strip()
        name = row['Name'].strip()
        lat  = float(row['Lat'])
        lon = float(row['Lon'])
        hgt = float(row['Hgt'])

        stations.append(WeatherStation(wmoid, name, lat, lon, hgt))

    return stations

def load_surface_stations_map(filename):
    stations = load_surface_stations(filename)
    return {station.name:station for station in stations}

################################################
# extract stations list based on Wyoming sondes
def extract_stations():
    print ("Extracting stations list...")

    sonde_dataset = WyomingSondeDataset()

    stations = {}

    files = []

    station_ids = []

    # collect all files in archive:
    for (dirpath, dirnames, filenames) in os.walk(archive_config.wyoming_sonde_dir):
        for filename in filenames:

            parts = filename.split("_")
            station_id = parts[2]
            if station_id not in station_ids:
                station_ids.append(station_id);

                (sonde_data, station) = sonde_dataset.read_sonde(dirpath + "/" + filename)

                stations[station.wmoid] = station

            files.extend(filename)
    return stations

#stations = { **extract_stations(), **stations }


def get_closest_station( stations, lat, lon):
    min_dist = -1
    closest_station = None

    for station in stations:
        dist = (lat - station.lat) ** 2 + (lon - station.lon) ** 2
        if dist < min_dist or min_dist < 0:
            min_dist = dist
            closest_station = station
    if(dist > 0.01) :
        print ('STATION MIN_DIST closest station', closest_station , min_dist)
    return closest_station

"""
Calulates a bounding box for stations list
:arg stations - list of station to include in returned AOI
:returns latmin, lonmin, latmax, lonmax
"""
def get_aoi(stations):
    for idx, station in enumerate(stations):
        if lonmin is None or lonmin > station.lon:
            lonmin = station.lon
        if lonmax is None or lonmax < station.lon:
            lonmax = station.lon
        if latmin is None or latmin > station.lat:
            latmin = station.lat
        if latmax is None or latmax < station.lat:
            latmax = station.lat

    lonmin = lonmin - 0.05
    lonmax = lonmax + 0.05
    latmin = latmin - 0.005
    latmax = latmax + 0.005

    return latmin, lonmin, latmax, lonmax

"""
Selects stations that are contanied in the provided AOI/bounding box 
"""
def filter_by_aoi(all_stations, latmin, lonmin, latmax, lonmax):

    stations = []
    for idx, station in enumerate(all_stations):
        if station.lon < lonmin or station.lon > lonmax:
            continue
        if station.lat < latmin or station.lat > latmax:
            continue
        stations.append(station)

    return stations