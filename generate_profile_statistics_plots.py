from vertical_profile import VerticalProfile
from plot_profile import plot_profile, plot_radial
import datasets.util as util
from  profile_database import ProfileDatabase
import datetime as dt
import numpy as np
import stations as stations_list
from datasets.wrf_dataset import WRFDataset
import datasets.beitdagan_sonde_dataset as beitdagan_sonde
import os
tag = 'config1'
outdir = f"plots/{tag}/profiles/"

# no of radiosonde: BD 40179
wmoId = 40179
station = stations_list.stations[wmoId]
stations = [ station ]

params = ["wvel_ms", "wdir_deg", "temp_k", "u_knt", "v_knt"]
#dt.datetime(2013, 7, 12, 18,00),
#dt.datetime(2013, 8, 12, 18,00)
#dt.datetime(2017,11, 25, 18,00)
#dt.datetime(2018, 2, 15, 18,00)
#dt.datetime(2018, 4, 30, 18,00)
dates = [
#dt.datetime(2013, 7, 12, 18,00)
#dt.datetime(2017, 9, 26, 18,00),
#dt.datetime(2017,11, 25, 18,00),
#dt.datetime(2018, 2, 15, 18,00),
#dt.datetime(2018, 4, 30, 18,00),
dt.datetime(2016, 10, 12, 18, 00),
dt.datetime(2018, 4, 29, 18, 00)

    #dt.datetime(2020, 9, 13, 18,00),
#dt.datetime(2020, 9, 14, 18,00),
#dt.datetime(2020, 9, 15, 18,00)
    ]

forecast_hours = [ 6, 18, 30, 42]
domains = ["d01", "d02", "d03", "d04"]

sonde_dataset = beitdagan_sonde.BeitDaganSondeDataset();

minh = 0
maxh = 5000
param_ranges={
    "Values":{"wvel_ms":(0, 25), "wdir_deg":(0, 400), "temp_k":(250, 310), "u_knt":(0, 50), "v_knt":(0,50)},
    "Bias":{"wvel_ms":(-15, 15), "wdir_deg":(-15, 15), "temp_k":(-10, 10), "u_knt":(-15, 15), "v_knt":(-15,15)},
    "RMSE":{"wvel_ms":(0, 15), "wdir_deg":(0, 25), "temp_k":(0, 25), "u_knt":(0, 15), "v_knt":(0,15)},
    "MAE":{"wvel_ms":(0, 15), "wdir_deg":(0, 25), "temp_k":(0, 25), "u_knt":(0, 15), "v_knt":(0,15)},
    "Variance":{"wvel_ms":(0, 15), "wdir_deg":(0, 25), "temp_k":(0, 25), "u_knt":(0, 15), "v_knt":(0,15)},
    "Expectation": {"wvel_ms":(0, 15), "wdir_deg":(0, 25), "temp_k":(0, 25), "u_knt":(0, 15), "v_knt":(0,15)}
}

base_wrf_dir = r"E:\meteo\urban-wrf\wrfout\\"
configs = ["bulk",  'bulk_sst' ]
db = ProfileDatabase()
all_datasets = {}
for config in configs:
    for domain in domains:
        ds_label = f"WRF {domain} {config}"
        dataset = WRFDataset(f"{base_wrf_dir}\\{config}", domain)
        all_datasets[ds_label] = dataset
        db.register_dataset(ds_label, dataset)
ref_profile_id = 'SONDE_HIRES'

def create_plots( date, forecast_offset, domain):

    start_date = date
    end_date = date

    dataset_labels = [ref_profile_id]
    for config in configs:
        dataset_labels.append(f"WRF {domain} {config}")
    config = 'bulk'

    datasets = []

    for label in dataset_labels:

        datasets.append(db.get_dataset(label, minh, maxh, params))



    # prepare arrays for statistics:
    heights = db.get_heights( minh, maxh)




    ####################################################
    # caching all relevant data:


    ref_profiles = {}
    all_profiles = {}
    #print("Caching profiles...")
    for (heights, ds_profiles, curr_date) in db.iterator(datasets, heights, station, start_date, end_date, forecast_offset):
        print("Extracting %s..." % curr_date)

        all_profiles[curr_date] = (heights, ds_profiles, curr_date)
        ref_profiles[curr_date] = ds_profiles[ref_profile_id]

    if len(all_profiles) == 0:
        print("No data found")
        exit()
    ####################################################
    # preparing arrays for statistics:

    all_bias = {}
    all_mae = {}
    all_rmse = {}

    all_mean = {}

    all_var = {}
    all_var2 = {}
    all_count = {}  # at each z level number of events  when model and measurement exist
    all_delta = {}
    for ds_label in dataset_labels:

        bias = all_bias[ds_label] = {}
        mae = all_mae[ds_label] = {}
        rmse = all_rmse[ds_label] = {}
        mean = all_mean[ds_label] = {}

        var = all_var[ds_label] = {}
        var2 = all_var2[ds_label] = {}
        count = all_count[ds_label] = {}
        delta = all_delta[ds_label] = {}

        for param in params:
            bias[param] = np.zeros((len(heights)))
            mae[param] = np.zeros((len(heights)))
            rmse[param] = np.zeros((len(heights)))
            mean[param] = np.zeros((len(heights)))

            var[param] = np.zeros((len(heights)))
            var2[param] = np.zeros((len(heights)))

            count[param] = np.zeros((len(heights)))
            delta[param] = np.zeros((len(heights)))
            delta[param][:] = np.nan

    #for (heights, profiles, curr_date) in all_profiles.values():
    #    for ds_label in iter(profiles):
    #        profile = profiles[ds_label]
     #       count = all_count[ds_label]
    #        for param in params:
    #            for ix in range(len(heights)):
    #                # data was found and this is not a missing value
    #                # for wind dir nan for missing data or when sp < 2 knt. wind dir values - are wrong
    #                # have to set correct values according to u v
    #                if (profile.values[param] is not None and not np.isnan(profile.values[param][ix])): count[param][ix] += 1

    ####################################################
    #  params = ["wvel_knt", "wdir_deg", "u_knt", "v_knt", "pres_hpa"]
    #   1 kt = 0.51444444444 mps

    #print("Calculating statistics...")
    for (heights, profiles, curr_date) in all_profiles.values():
        sonde = ref_profiles[curr_date]
        for ds_label in iter(profiles):
            model = profiles[ds_label]

            bias = all_bias[ds_label]
            mae = all_mae[ds_label]
            rmse = all_rmse[ds_label]
            mean = all_mean[ds_label]

            var = all_var[ds_label]
            var2 = all_var2[ds_label]
            count = all_count[ds_label]
            delta = all_delta[ds_label]

            for param in params:
                model_values = model.values[param]  # type: np array
                sonde_values = sonde.values[param]  # type: np array
                # print param
                # print model_values
                # print sonde_values
                if model_values is None or sonde_values is None:
                    continue
                delta = np.zeros((len(heights)))
                delta[:] = np.nan
                if param == "wdir_deg":
                    for ix in range(len(heights)):
                        if model_values[ix] is not np.nan and sonde_values[ix] is not np.nan:
                            delta[ix] = util.to_degrees( \
                                                    mean["u_knt"][ix]-model.values["u_knt"][ix], \
                                                    mean["v_knt"][ix]- model.values["v_knt"][ix])
                            #delta[ix] = util.to_degrees(sonde.values["u_knt"][ix], sonde.values["v_knt"][ix]) \
                            #            - \
                            #            util.to_degrees(model.values["u_knt"][ix], model.values["v_knt"][ix])

                            delta[ix] = util.wrap_degrees(delta[ix])
                            if delta[ix] > 180:
                                delta[ix] = delta[ix]-360

                else:
                    for ix in range(len(heights)):
                        if model_values[ix] is not np.nan and sonde_values[ix] is not np.nan:
                            delta[ix] = sonde_values[ix] - model_values[ix]

                for ix in range(len(heights)):
                    if (not np.isnan(delta[ix])): count[param][ix] += 1
                    if (np.isnan(delta[ix])): delta[ix] = 0  # delta = 0 , do not increase number of events
                # print count[param]

                bias[param] += delta

                for ix in range(len(heights)):
                    #if not np.isnan(sonde_values[ix]):
                        mean[param][ix] += model_values[ix]

                mae[param] += abs(delta)
                rmse[param] += delta ** 2

            for param in params:
                if param != "wdir_deg":
                    for ix in range(len(heights)):
                        if count[param][ix] != 0:
                            mean[param][ix] /= count[param][ix]

            # print sonde_mean[param]
            for ix in range(len(heights)):
                if count[param][ix] != 0:
                    bias[param][ix] /= count[param][ix]
                    mae[param][ix] /= count[param][ix]
                    rmse[param][ix] = (rmse[param][ix] / count[param][ix]) ** 0.5

            for ix in range(len(heights)):
                if (not np.isnan(mean["u_knt"][ix])):
                    mean["wdir_deg"][ix] = util.to_degrees(mean["u_knt"][ix], mean["v_knt"][ix])

    # completed mean bias ame rmse calculations
    # print sonde_mean["wdir_deg"]


    ####################################################
    # second pass: calculate variance


    for (heights, profiles, curr_date) in all_profiles.values():
        sonde = ref_profiles[curr_date]
        #sonde = ref_ds.get_profile(curr_date, forecast_offset, station)
        for ds_label in iter(profiles):
            model = profiles[ds_label]

            bias = all_bias[ds_label]
            mae = all_mae[ds_label]
            rmse = all_rmse[ds_label]
            mean = all_mean[ds_label]

            var = all_var[ds_label]
            var2 = all_var2[ds_label]
            count = all_count[ds_label]
            delta = all_delta[ds_label]

            for param in params:

                model_values = model.values[param]  # type: np array
                sonde_values = sonde.values[param]  # type: np array
                if model_values is None or sonde_values is None:
                    continue

                if param == "wdir_deg":  # define dir values from u v  again  model_values, sonde_values not relevant for wind dir
                    for ix in range(len(heights)):
                        if mean[param][ix] is not np.nan and model_values[ix] is not np.nan:
                            delta[param][ix] = util.to_degrees( \
                                                    mean["u_knt"][ix]-model.values["u_knt"][ix], \
                                                    mean["v_knt"][ix]- model.values["v_knt"][ix])
                        if delta[param][ix] > 180:
                            delta[param][ix] = 360-delta[param][ix]
                         # print sonde_delta[param][ix]

                    for ix in range(len(heights)):
                        if np.isnan(delta[param][ix]):
                            # skip assigment,
                            # in case of a single day, we will get 0 error- which means - no data , so one can not tell whether its a perfect
                            # match between model and rad or no data case
                            delta[param][ix] = 0
                        else:
                            delta[param][ix] = np.abs(util.wrap_degrees(delta[param][ix]))


                else:

                    for ix in range(len(heights)):
                        if not np.isnan(model_values[ix]) and not np.isnan(mean[param][ix]):
                            delta[param][ix] = mean[param][ix] - model_values[ix]
                        else:
                            delta[param][ix] = 0

                        # second term for wind dir variance calculations:
                var2[param] += delta[param]
                var[param] += delta[param] ** 2

            # finalize computation:
            # calculate variance:
            for param in params:
                if param == "wdir_deg":
                    for ix in range(len(heights)):
                        if count[param][ix] != 0.:
                            var[param][ix] = (var[param][ix] / count[param][ix] - (
                                        var2[param][ix] / count[param][ix]) ** 2) ** 0.5

                else:

                    for ix in range(len(heights)):
                        if count[param][ix] != 0.:
                            var[param][ix] = (var[param][ix] / count[param][ix]) ** 0.5

    # print number of events :
    #print('number of days [wdir] = ', count["wdir_deg"], count["wdir_deg"], count["wdir_deg"])
    '''
    # print results, screen and file:
    ofile = outdir + 'output_statistics_wind'

    of = open(ofile, 'w')
    for idx in range(len(bias["wvel_ms"])):
        of.writelines("%6dm : bias:%3.3f mae:%3.3f rmse:%3.3f \n" % (
            heights[idx], bias["wvel_ms"][idx], mae["wvel_ms"][idx], rmse["wvel_ms"][idx]))

        print("%6dm : bias:%3.3f mae:%3.3f rmse:%3.3f" % (
            heights[idx], bias["wvel_ms"][idx], mae["wvel_ms"][idx], rmse["wvel_ms"][idx]))
    '''

    all_bias[ref_profile_id] = None
    all_rmse[ref_profile_id] = None
    all_mae[ref_profile_id] = None
    all_var[ref_profile_id] = None
    all_var2[ref_profile_id] = None

    draw_array(datasets, config, heights, all_mean, start_date, forecast_offset, "Values")
    draw_array(datasets, config, heights, all_bias, start_date, forecast_offset, "Bias")
    draw_array(datasets, config, heights, all_rmse, start_date, forecast_offset, "RMSE")
    draw_array(datasets, config, heights, all_mae, start_date, forecast_offset, "MAE")
    draw_array(datasets, config, heights, all_var, start_date, forecast_offset, "Variance")
    draw_array(datasets, config, heights, all_var2, start_date, forecast_offset, "Expectation")

    ########################################

    # dont display edges of interpolation results
    # suffer from low number of points
    # radiosonde dont have a lot of points at the higher levels 20-24 km
    # therefore nf=len(heights)-2
    # draw from index 2 , eg mae[draw_param][2: nf]
    #

def draw_array(datasets, config, heights, values, start_date, forecast_offset, values_tag):
    for draw_param in ["wdir_deg", "wvel_ms", "temp_k"]:
        nf = len(heights) - 2

        xlim = param_ranges[values_tag][draw_param]
        series = {}
        for dataset in datasets:
            profiles = values[dataset.dataset_label]
            if profiles is not None:
                series[dataset.dataset_label] = profiles[draw_param][0: nf]

        plot_outdir = f"{outdir}/{start_date.strftime('%Y%m%d%H')}/{domain}/{forecast_offset:02d}/{values_tag}/"
        os.makedirs(plot_outdir, exist_ok=True)
        prefix = f'vertical_profile_{config}_{values_tag}_{start_date.strftime("%Y%m%d%H")}Z+{forecast_offset:02d}_{domain}_{draw_param}'

        title = f"{values_tag} {draw_param} {start_date}Z+{forecast_offset:02d}h, station {station.wmoid}"

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


total_plots = len(dates) * len(forecast_hours) * len(domains)
plot_idx = 1
for date in dates:
    for foffset in forecast_hours:
        for domain in domains:
            print(f"Plotting {plot_idx}/{total_plots}) {date} {foffset} {domain}...")
            plot_idx += 1
            means = create_plots(date, foffset, domain)
