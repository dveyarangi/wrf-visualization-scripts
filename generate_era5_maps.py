import xarray as xr
ds = xr.open_dataset(r'E:\meteo\urban-wrf\era5\2017\era5_20170926_an_sfc_0.grib', engine='cfgrib')

#dataset = util.load_dataset(r'E:\meteo\urban-wrf\era5\2017\era5_20170926_an_sfc_0.grib')

print(ds)