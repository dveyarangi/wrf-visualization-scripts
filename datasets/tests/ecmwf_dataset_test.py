import datetime as dt
from python.datasets.ecmwf_dataset import ECMWFDataset


################################################
# TEST

dataset = ECMWFDataset()

datetime = dt.datetime(2016,07,01,00,00)

params = [ "wvel_knt", "wdir_deg", "u_knt", "v_knt"]

profile = dataset.get_profile(32,32,datetime, 20000,30000, params)

#profile = dataset.get_profile(32,32,datetime, 20000,30000, params)

print( profile )