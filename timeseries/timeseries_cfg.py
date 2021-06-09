import datetime as dt
import stations as st

runtag = ''
outdir = f'D:\\Dev\\Machon\\Urban\\plots{runtag}\\'

time_groups = [
            (dt.datetime(2013, 7, 12, 18, 00), dt.datetime(2013, 7, 14, 18, 00)), \
            (dt.datetime(2016, 10, 12, 18, 00), dt.datetime(2016, 10, 15, 00, 00)),
            (dt.datetime(2017, 9, 26, 18, 00),dt.datetime(2017, 9, 28, 18, 00)), \
            (dt.datetime(2017, 11, 25, 18, 00),dt.datetime(2017, 11, 27, 18, 00)), \
            (dt.datetime(2018, 2, 15, 18, 00),dt.datetime(2018, 2, 17, 18, 00)),
            (dt.datetime(2018, 4, 30, 18, 00),dt.datetime(2018, 5, 2, 18, 00)), \
            (dt.datetime(2020, 9, 13, 18, 00),dt.datetime(2020, 9, 15, 18, 00)), \
            (dt.datetime(2020, 9, 14, 18, 00),dt.datetime(2020, 9, 16, 18, 00)), \
            (dt.datetime(2020, 9, 15, 18, 00),dt.datetime(2020, 9, 17, 18, 00)), \
            #(dt.datetime(2018, 4, 29, 18, 00), dt.datetime(2018, 5, 2, 18, 00))
]

tags = ['config1']

#print(obs_series)

params = ["wvel_ms", "wdir_deg", "temp2m_k", "u10_ms", "v10_ms", "rh2m"]
#params = ["temp2m_k"]
param_ranges = {
"wvel_ms":(0,10), "wdir_deg":(0, 150), "temp2m_k":(0,10), "rh2m": (0,50)
}
domain_timestep = {"d04":20}
configs = {'bulk_ysu':'bulk_ysu','bulk_sst':'bulk','slucm':'slucm'}
#configs = {'bulk_ysu':'bulk_ysu'}
time_range_groups = [ (10,22),(22,34),(34,48) ]

station_names = [ \

                 'Afek', \
                 'Ein Karmel', \
                 #'Haifa Refineries', \
                 'Haifa Technion', \
                 'Haifa University', \
                 'Shavei Zion',
                  \
'Nesher','K.HAIM-REGAVIM','K.Hasidim','K.Bnyamin','K.Tivon','K.Yam','Shprinzak','N.Shanan','Ahuza','K.Bialik-Ofarim','IGUD','K.Ata'
                ]
'''
station_names.extend( [

    "Ein Karmel North",
    "Haifa Technion West",
    "Haifa University South",
    "Haifa University West",
    "Shavei Zion East",
    "Nesher South",
    "K.Hasidim East",
    "N.Shanan South",
    "N.Shanan West",
    "Ahuza East",
    "K.Bialik-Ofarim South",
    "K.Bialik-Ofarim East",
    "K.Ata East",
])
'''
all_stations = st.load_surface_stations_map('../etc/stations_and_neighbors.csv')
stations = []
for station_name in station_names:
    stations.append(all_stations[station_name])
time_range_groups = [
    (10,22),(22,34),(34,48)
]

#time_range_groups = [ (x,x+3) for x in range(6,48,3)]


domain_groups = [["d04"]]
