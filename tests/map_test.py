import stations as st

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
stations = []


lon_min = 34.6
lon_max = 35.6
lat_min = 31.9
lat_max = 32.75

for station_name in station_names:
    station = st.stations[station_name]
    if station.lon >= lon_min and station.lon <= lon_max and station.lat >= lat_min and station.lat >= lat_max:
        print(station_name)
    stations.append(st.stations[station_name])


