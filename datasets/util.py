from netCDF4 import Dataset
import datetime as dt
import numpy as np
import math
import pytz
import os
os.environ['PROJ_LIB'] = 'D:/Dev/Python/anaconda3/envs/Urban/Library/share/basemap'


def load_dataset( path):
    try:
        # open netcdf dataset:
        return Dataset(path)

    except (OSError, IOError) as e:  # failed to
        print ("Failed to open ", path, " : ", e)
        return None


# convert string to float, return None if failed
def to_float(value_str):
    try:
        return float(value_str)
    except ValueError:
        return None

epoch = dt.datetime.utcfromtimestamp(0)

def unix_time_millis(dt):
    return (dt - epoch).total_seconds() * 1000.0

epoch_localized = pytz.UTC.localize(dt.datetime.utcfromtimestamp(0))

def unix_time_millis_localized(dt):
    return (dt - epoch_localized).total_seconds() * 1000.0

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + dt.timedelta(n)


def to_degrees(u, v):
    return (270. - (np.arctan2(v, u)*(180./math.pi))) % 360.


def wrap_degrees(deg):
    if deg > 360:
        deg -= 360
    if deg < 0:
        deg += 360
    return deg


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]
