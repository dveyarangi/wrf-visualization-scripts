#!/usr/bin/env python
import datetime as dt
import numpy as np
import os.path

from profile_database import ProfileDatabase
from vertical_profile import VerticalProfile
from python.plot_profile import plot_profile
import stations as stations_list

################################################
# TEST


# parameters:

# no of radiosonde:
wmoId = 40179
station = stations_list.stations[wmoId]
stations = [ station ]
minh = 9000
maxh = 25000

# pick compared datasets:
#
# TODO: comparing sondes won't work, since they have variable size
model_label = "WRF" # WRF; TODO: ECMWF doesn't work yet
sonde_label = "LORES" # LORES or HIRES

# TODO: only those params are currently available:
params = ["wvel_knt", "wdir_deg", "u_knt", "v_knt"]
# param = "wdir_deg"

# date ranges:
start_date = dt.datetime(2016, 01, 20,00,00)
end_date   = dt.datetime(2016, 01, 30,00,00)

# output figs dir:
outdir='/home/sigalit/Loon/machon-winds-master/'+str(wmoId)+'_'+sonde_label+'_Apr_2016_statistics/'
if not os.path.exists(outdir):
    os.makedirs(outdir)


# fetch data handles:
db = ProfileDatabase()
model_ds = db.get_dataset(model_label, minh, maxh, params)
sonde_ds = db.get_dataset(sonde_label, minh, maxh, params)


# prepare arrays for statistics:
heights = db.get_heights( minh, maxh)

rad_dir="//home/sigalit/sigdata/Loon_radiosondes/"


####################################################
# caching all relevant data:

profiles = {}
print("Caching profiles...")
for (heights, model, sonde, curr_date) in db.iterator(model_ds, sonde_ds, heights, station, start_date, end_date):
    print("Extracting %s..." % curr_date)
    profiles[curr_date] = (heights, model, sonde, curr_date)


####################################################
# preparing arrays for statistics:

count = 0
bias = {}
mae = {}
rmse = {}
model_mean = {}
sonde_mean = {}
model_var = {}
sonde_var = {}
for param in params:
    bias[param] = np.zeros((len(heights)))
    mae[param] = np.zeros((len(heights)))
    rmse[param] = np.zeros((len(heights)))
    model_mean[param] = np.zeros((len(heights)))
    sonde_mean[param] = np.zeros((len(heights)))

    model_var[param] = np.zeros((len(heights)))
    sonde_var[param] = np.zeros((len(heights)))

####################################################
print("Calculating statistics...")
for (heights, model, sonde, curr_date) in profiles.values():
    for param in params:
        model_values = model.values[param]  # type: np array
        sonde_values = sonde.values[param]  # type: np array
        if model_values is None or sonde_values is None:
            continue

        if param == "wdir_deg":
            delta = 180 - np.abs(np.abs(model_values - sonde_values) - 180)
        else:
            delta = sonde_values - model_values

        bias[param] += delta

        model_mean[param] += model_values
        sonde_mean[param] += sonde_values

        mae[param] += abs(delta)
        rmse[param] += delta**2

    count = count + 1

for param in params:
    model_mean[param] /= count
    sonde_mean[param] /= count

####################################################
# second pass:
count=0
for (heights, model, sonde, curr_date) in profiles.values():

    for param in params:
        model_values = model.values[param]  # type: np array
        sonde_values = sonde.values[param]  # type: np array
        if model_values is None or sonde_values is None:
            continue

        if param == "wdir_deg":
            model_delta = 180 - np.abs(np.abs(model_mean[param] - sonde_values) - 180)
            sonde_delta = 180 - np.abs(np.abs(sonde_mean[param] - sonde_values) - 180)
        else:
            model_delta = model_mean[param] - model_values
            sonde_delta = sonde_mean[param] - sonde_values

        model_var[param] += model_delta**2
        sonde_var[param] += sonde_delta**2
    count = count + 1

# finalize computation:
for param in params:
    bias[param] /= count
    mae[param] /= count
    rmse[param] = (rmse[param] / count)**0.5

    model_var[param] = (model_var[param] / count)**0.5
    sonde_var[param] = (sonde_var[param] / count)**0.5


wdir_deg_model_mean = 270-np.rad2deg(np.arctan(model_mean["v_knt"]/model_mean["u_knt"]))
wdir_deg_sonde_mean = 270-np.rad2deg(np.arctan(sonde_mean["v_knt"]/sonde_mean["u_knt"]))

# print number of events :
print 'number of days = ',count
# print results:
for idx in range(len(bias["wvel_knt"])):

    print("%6dm : bias:%3.3f mae:%3.3f rmse:%3.3f" % (heights[idx], bias["wvel_knt"][idx], bias["wvel_knt"][idx], bias["wvel_knt"][idx]))

########################################


for draw_param in params:
    title = "Errors for %s, station %s" % (curr_date, station.wmoid)
    plot_profile(
        VerticalProfile(heights,{
          draw_param + " MAE": mae[draw_param],
          draw_param + " RMSE": rmse[draw_param]
        }, station),
        VerticalProfile(heights, {
          draw_param + " bias": bias[draw_param]
        }, station),
        outdir, title)
    # save figure

    title = "Means for %s to %s, station %s" % (start_date, end_date, station.wmoid)
    plot_profile(
        VerticalProfile(heights, {
           draw_param + " model mean": model_mean[draw_param],
           draw_param + " sonde mean": sonde_mean[draw_param]
        }, station),
        None, outdir, title)

    title = "Variance for %s to %s, station %s" % (start_date, end_date, station.wmoid)
    plot_profile(
        VerticalProfile(heights, {
           draw_param + " model variance": model_var[draw_param],
           draw_param + " sonde variance": sonde_var[draw_param]
        }, station),
        None, outdir, title)

plot_profile(
    VerticalProfile(heights, {
       "Model angular dir": model_mean["wdir_deg"],
       "Model UV dir": wdir_deg_model_mean,
        "Sonde angular dir": sonde_mean["wdir_deg"],
        "Sonde UV dir": wdir_deg_sonde_mean
    }, station),
    None, outdir, "")

