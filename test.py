import xarray as xr
import cfgrib

cfgrib.check()

filename = r'D:\Dev\Machon\Urban\tests\era5-levels-members.grib'
sample_ds = xr.open_dataset(filename, engine='cfgrib')

print( sample_ds)