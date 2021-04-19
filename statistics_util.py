import numpy as np

def get_delta_series(model, surface, param):

    model_values = model.values[param]  # type: nparray
    surface_values = surface.values[param]  # type: nparray

    if model_values is None or surface_values is None:
        return None

    delta = np.zeros(len(model.xs))
    delta[:] = np.nan
    for ix, xtime in enumerate(model.xs):

        if np.isnan(model_values[ix]) or np.isnan(surface_values[ix]):
            delta[ix] = 0  # delta = 0 , do not increase number of events
            continue
        if param == "wdir_deg":
            su = surface.values["u10_ms"][ix]
            sv = surface.values["v10_ms"][ix]
            if (su ** 2 + sv ** 2) >= 1:
                mu = model.values["u10_ms"][ix]
                mv = model.values["v10_ms"][ix]
                ds = util.to_degrees(su, sv)
                dm = util.to_degrees(mu, mv)
                diff = dm - ds
                source_angle_diff = (diff + 180) % 360 - 180
                delta[ix] = source_angle_diff

        elif param == "wvel_ms":
            su = surface.values["u10_ms"][ix]
            sv = surface.values["v10_ms"][ix]
            if (su ** 2 + sv ** 2) >= 1:
                delta[ix] = model_values[ix] - surface_values[ix]
        else:
            delta[ix] = model_values[ix] - surface_values[ix]

        if np.isnan(delta[ix]):
            continue