import numpy as np
import math
class Series:

    def __init__(self, xs, vals, station, angular=[]):
        self.xs = xs
        self.values = vals
        self.station = station

        for param in iter(vals):
            if param in angular:
                radians = np.deg2rad(vals[param])
                unwrapped_radians = np.copy(radians)
                unwrapped_radians[~np.isnan(radians)] = np.unwrap(radians[~np.isnan(radians)])

                degrees = np.rad2deg(unwrapped_radians)
                vals[param] = degrees

    # given a list of x values,
    # interpolate or integrate sonde data to them
    def rescale(self, nxs):

        rescaled_values = np.zeros((len(nxs)))

        for i in range(0, len(nxs) - 0):

            # pick range for integration:
            low_level = None
            high_level = None
            curr_level = nxs[i]
            if i == 0:
                low_level = curr_level
            else:
                low_level = (nxs[i - 1] + curr_level) / 2

            if i == len(nxs) - 1:
                high_level = curr_level
            else:
                high_level = (nxs[i + 1] + curr_level) / 2

            range_samples = []
            for (idx, hgt) in enumerate(self.xs):
                if high_level >= hgt >= low_level:
                    range_samples.append(self.values[idx])

            if len(range_samples) >= 2:
                rescaled_values[i] = self.average(range_samples)
            else:
                rescaled_values[i] = self.interp(curr_level)

        return Series(nxs, rescaled_values, self.station)

    def average(self, samples):

        return sum(samples) / len(samples)

    def interp(self, x):
        x0 = None
        y0 = None
        x1 = None
        y1 = None
        for (idx, hgt) in enumerate(self.xs):
            value = self.values[idx]
            if hgt == x:
                return value
            if hgt > x:
                x0 = hgt
                y0 = value
            else:
                x1 = hgt
                y1 = value
                break

        if x0 is None and x1 is None: return None
        if x0 is None: return y1
        if x1 is None: return y0

        y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)

        return y

    def interpolate(self, nxs):
        interpvals = {}
        for param in iter(self.values):
            skip = 0
            for il in range(len(self.xs)):
                if np.isnan(self.values[param][il]):
                    skip = skip + 1

            if skip >= (len(self.xs)) * 0.75:
                interpvals[param] = np.zeros(len(self.xs))
                interpvals[param][:] = np.nan
            else:
                interpvals[param] = np.interp(nxs, self.xs, self.values[param], left=np.nan,
                                              right=np.nan)

        return Series(nxs, interpvals, self.station)

    def moving_average(self, x, w):
        return np.convolve(x, np.ones(w), 'same') / w

    def _shift(self, arr, num, fill_value=np.nan):
        arr = np.roll(arr, num)
        if num < 0:
            arr[num:] = fill_value
        elif num > 0:
            arr[:num] = fill_value
        return arr
    def shift(self, steps_right):
        values = {}
        for param in iter(self.values):
            skip = 0
            for il in range(len(self.xs)):
                if np.isnan(self.values[param][il]):
                    skip = skip + 1

            if skip >= (len(self.xs)) * 0.75:
                values[param] = np.zeros(len(self.xs))
                values[param][:] = np.nan
            else:
                shifted_arr = self._shift(self.values[param], steps_right)
                values[param] = shifted_arr

        #xs = self._shift(self.xs, steps_right)
        return Series(self.xs, values, self.station)

    def integrate(self, window):
        interpvals = {}
        for param in iter(self.values):
            skip = 0
            for il in range(len(self.xs)):
                if np.isnan(self.values[param][il]):
                    skip = skip + 1

            if skip >= (len(self.xs)) * 0.75:
                interpvals[param] = np.zeros(len(self.xs))
                interpvals[param][:] = np.nan
            else:
                if param == "wdir_deg":
                    u10avg = self.moving_average(self.values["u10_ms"], window)
                    v10avg = self.moving_average(self.values["v10_ms"], window)
                    interpvals[param] = (270. - ( np.arctan2(v10avg,u10avg)*(180./math.pi) ))%360.
                else:
                    interpvals[param] = self.moving_average(self.values[param], window)

        return Series(self.xs, interpvals, self.station)

