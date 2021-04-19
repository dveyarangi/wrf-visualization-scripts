

class SpatialIndex:

    def __init__(self):

        self.points = []

    # add an indexed point in format (lat, lon, i, j)
    def add(self, point):
        self.points.append( point )

    def add(self, lat, lon, i, j):
        self.points.append( (lat, lon, i, j) )

    def get_closest_index(self, lat, lon):

        min_dist = -1
        ci = -1
        cj = -1
        clat = -1
        clon = -1

        for (p_lat, p_lon, i, j) in self.points:
            dist = (lat - p_lat) ** 2 + (lon - p_lon) ** 2
            if dist < min_dist or min_dist < 0:
                ci = i
                cj = j
                clat = p_lat
                clon = p_lon
                min_dist = dist
        if min_dist > 0.003:
            print('STATION MIN_DIST lat lon', min_dist, lat, clat, lon, clon )
        return ci, cj, clat, clon
