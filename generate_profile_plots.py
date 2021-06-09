from vertical_profile import VerticalProfile
from plot_profile import plot_profile, plot_radial
import datasets.util as util
from  profile_database import ProfileDatabase
import datetime as dt
import numpy as np
import stations as st
from datasets.wrf_dataset import WRFDataset
import datasets.beitdagan_sonde_dataset as beitdagan_sonde
import os
tag = 'config1'
outdir = f"plots/{tag}/profiles/"

# no of radiosonde: BD 40179
wmoId = 40179

all_stations = st.load_surface_stations_map('etc/stations.csv')
station = all_stations['Beit Dagan']
stations = [ station ]

params = ["wvel_ms", "wdir_deg", "temp_k"]
#dt.datetime(2013, 7, 12, 18,00),
#dt.datetime(2013, 8, 12, 18,00)
#dt.datetime(2017,11, 25, 18,00)
#dt.datetime(2018, 2, 15, 18,00)
#dt.datetime(2018, 4, 30, 18,00)
dates = [
#dt.datetime(2013, 7, 12, 18,00)
    (dt.datetime(2016, 10, 12, 18, 00), [ 6, 18, 30, 42 ]),
    (dt.datetime(2017,  9, 26, 18, 00), [ 6, 18, 30, 42 ]),
    (dt.datetime(2017, 11, 25, 18, 00), [ 6, 18, 30, 42 ]),
    (dt.datetime(2018,  2, 15, 18, 00), [ 6, 18, 30, 42 ]),
    (dt.datetime(2018,  4, 30, 18, 00), [ 6, 18, 30, 42 ]),

#    (dt.datetime(2018, 4, 29, 18, 00), [ 6, 18, 30, 42, 54, 66])

    #dt.datetime(2020, 9, 13, 18,00),
#dt.datetime(2020, 9, 14, 18,00),
#dt.datetime(2020, 9, 15, 18,00)
    ]

domains = ["d01", "d02", "d03", "d04"]

sonde_dataset = beitdagan_sonde.BeitDaganSondeDataset();

minh = 0
maxh = 2700
param_ranges={
    "Values":{"wvel_ms":(0, 25), "wdir_deg":(0, 400), "temp_k":(250, 310)},
 }

base_wrf_dir = r"E:\meteo\urban-wrf\wrfout\\"
configs = {'bulk_sst':'bulk', 'slucm':'slucm','bulk_ysu':'bulk_ysu'}
db = ProfileDatabase()

all_datasets = {}
for config in configs:
    for domain in domains:
        ds_label = f"WRF {domain} {config}"
        dataset = WRFDataset(f"{base_wrf_dir}\\{config}", domain)
        all_datasets[ds_label] = dataset
        db.register_dataset(ds_label, dataset)
ref_profile_id = 'SONDE_HIRES'

domain_time_offsets = {'d01': [-3,0, +3], 'd02':range(-3, +4), 'd03':range(-3, +4), 'd04':range(-3, +4)}

def create_plots( date, forecast_offset, domain):

    start_date = date
    end_date = date

    dataset_labels = []
    for config in configs:
        dataset_labels.append(f"WRF {domain} {config}")
    config = 'bulk'

    datasets = []

    for label in dataset_labels:

        datasets.append(db.get_dataset(label, minh, maxh, params))

    ref_dataset = db.get_dataset(ref_profile_id, minh, maxh, params)

    # prepare arrays for statistics:
    heights = db.get_heights( minh, maxh)




    ####################################################
    # caching all relevant data:


    ref_profiles = {}
    all_profiles = {}
    #print("Caching profiles...")
    for foffset in domain_time_offsets[domain]:
        for (heights, ds_profiles, curr_date) in db.iterator(datasets, heights, station, start_date, end_date, foffset+forecast_offset):
            print("Extracting %s..." % curr_date)
            all_profiles[f'{foffset+forecast_offset}' ] = (heights, ds_profiles, curr_date)

    for (heights, ds_profiles, curr_date) in db.iterator([ref_dataset], heights, station, start_date, end_date, forecast_offset):
        ref_profiles[curr_date] = ds_profiles[ref_profile_id]

    if len(all_profiles) == 0:
        print("No data found")
        exit()
    ####################################################

    all_values = {}


    for config in configs:

        ds_label = f"WRF {domain} {config}"
        for key in all_profiles:
            (heights, profiles, curr_date) = all_profiles[key]
            sonde = ref_profiles[curr_date]
            sonde_values = all_values['OBS'] = {}
            for param in params:

                sonde_values[param] = np.zeros((len(heights)))
                if sonde_values is None:
                    continue

                for ix in range(len(heights)):
                    sonde_values[param][ix] = sonde.values[param][ix]

            values = all_values[f'+{key}h'] = {}
            for param in params:
                values[param] = np.zeros((len(heights)))

            model = profiles[ds_label]

            for param in params:
                model_values = model.values[param]  # type: nparray

                if model_values is None:
                    continue

                for ix in range(len(heights)):
                    values[param][ix] = model_values[ix]

    # completed mean bias ame rmse calculations
    # print sonde_mean["wdir_deg"]


        draw_array(configs[config], heights, all_values, start_date, forecast_offset, "Values")

    ########################################

    # dont display edges of interpolation results
    # suffer from low number of points
    # radiosonde dont have a lot of points at the higher levels 20-24 km
    # therefore nf=len(heights)-2
    # draw from index 2 , eg mae[draw_param][2: nf]
    #

def draw_array(config, heights, values, start_date, forecast_offset, values_tag):
    for draw_param in ["wdir_deg", "wvel_ms", "temp_k"]:
        nf = len(heights) - 2

        xlim = param_ranges[values_tag][draw_param]
        xlim = None
        series = {}
        for dataset_label in values:
            profiles = values[dataset_label]
            if profiles is not None:
                series[dataset_label] = profiles[draw_param][0: nf]

        plot_outdir = f"{outdir}/{start_date.strftime('%Y%m%d%H')}/{domain}/{forecast_offset:02d}/{values_tag}/"
        os.makedirs(plot_outdir, exist_ok=True)
        prefix = f'vertical_profile_{config}_{values_tag}_{start_date.strftime("%Y%m%d%H")}+{forecast_offset:02d}_{domain}_{draw_param}'

        title = f"WRF {domain} {config} {draw_param} {start_date.strftime('%Y%m%d %H')}Z+{forecast_offset:02d}h, {station.wmoid}"

        if "wdir_deg" == draw_param:
            plot_radial(
                VerticalProfile(heights[0: nf], series, station, True),
                None, plot_outdir, xlim, title, prefix + "_rad")
            plot_profile(
                VerticalProfile(heights[0: nf], series, station, True),
                None, plot_outdir, xlim, title, prefix )
        else:
            plot_profile(
                VerticalProfile(heights[0: nf], series, station),
                None, plot_outdir, xlim, title, prefix )

total_plots = 0
for (date, forecast_hours) in dates:
    total_plots += len(forecast_hours)
total_plots *= len(domains)
plot_idx = 1
for (date, forecast_hours) in dates:
    for foffset in forecast_hours:
        for domain in domains:
            print(f"Plotting {plot_idx}/{total_plots}) {date} {foffset} {domain}...")
            plot_idx += 1
            means = create_plots(date, foffset, domain)
