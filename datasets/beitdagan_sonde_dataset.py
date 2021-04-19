
from datasets.datasets import ProfileDataset
from datasets.beitdagan_sonde_edt_dataset import BeitDaganSondeEDTDataset
from datasets.beitdagan_sonde_ptu_dataset import BeitDaganSondePTUDataset
from datasets.beitdagan_sonde_physical_dataset import BeitDaganSondePhysicalDataset


################################################
# this class provides access to Beit Dagan
# high resolution sonde dataset
#
class BeitDaganSondeDataset(ProfileDataset):

    def __init__(self):
        self.edt_dataset = BeitDaganSondeEDTDataset()
        self.phy_dataset = BeitDaganSondePhysicalDataset()
        self.ptu_dataset = BeitDaganSondePTUDataset()

    def get_station_profile(self, station, datetime, forecast_hours, minh, maxh, params):

        if datetime.year < 2014:
            return self.phy_dataset.get_station_profile(station, datetime, forecast_hours, minh, maxh, params)
        elif datetime.year < 2017:
            return self.ptu_dataset.get_station_profile(station, datetime, forecast_hours, minh, maxh, params)
        else:
            return self.edt_dataset.get_station_profile(station, datetime, forecast_hours, minh, maxh, params)