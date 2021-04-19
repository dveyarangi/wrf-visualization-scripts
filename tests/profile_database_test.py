from profile_database import ProfileDatabase
import datetime as dt
import stations as stations_list
from plot_profile import plot_profile
from vertical_profile import VerticalProfile

################################################
# TEST

test_date = dt.datetime(2016,7,1,00,00)

wmoId = 40179
station = stations_list.stations[wmoId]
stations = [ station ]
minh = 15000
maxh = 25000
params = ["wvel_knt", "wdir_deg", "u_knt", "v_knt", "pres_hpa"]
#param = "wdir_deg"

db = ProfileDatabase()
wrf_profile = db.get_profile("WRF", station, test_date, minh, maxh, params)

hi_sonde_profile = db.get_profile("HIRES", station, test_date, minh, maxh, params)

lo_sonde_profile = db.get_profile("LORES", station, test_date, minh, maxh, params)


rescaled_hi_profile = hi_sonde_profile.interpolate(wrf_profile.heights)
rescaled_lo_profile = lo_sonde_profile.interpolate(wrf_profile.heights)




plot_profile(
    VerticalProfile(wrf_profile.heights,
        {   "WRF WD(knot)" : wrf_profile.values["wvel_knt"],
            "HIRES WD(knot)" : rescaled_hi_profile.values["wvel_knt"],
            "LORES WD(knot)" : rescaled_lo_profile.values["wvel_knt"]}, station),
        None, None, title="")