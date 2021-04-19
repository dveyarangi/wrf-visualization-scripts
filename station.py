class WeatherStation:

    def __init__(self, wmoid, name, lat, lon, hgt):
        self.wmoid = wmoid
        self.name = name
        self.lat = lat
        self.lon = lon
        self.hgt = hgt


    def __str__(self):
        return "WeatherStation (wmoid:%s, lat:%3f, lon:%3f, hgt:%3f)" % (self.wmoid, self.lat, self.lon, self.hgt)

