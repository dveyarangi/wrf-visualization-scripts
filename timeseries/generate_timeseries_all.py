from timeseries.generate_surface_timeseries_statistics_plots import generate as generate_station_event_statistics
from timeseries.generate_surface_timeseries_plots import generate as generate_station_event_plots
from timeseries.generate_surface_timeseries_avg_plots import generate as generate_stationma_event_plots
from timeseries.generate_surface_timeseries_correlation_plots import generate as generate_station_event_corr
from timeseries.generate_surface_timeseries_allstations_statistics_plots import generate as generate_allstations_event_statistics
from timeseries.generate_surface_timeseries_eventavg_statistics_plots import generate as generate_station_eventavg_statistics
from timeseries.generate_surface_timeseries_stationavg_statistics_plots import generate as generate_stationavg_event_statistics
from timeseries.generate_surface_timeseries_allavg_statistics_plots import generate as generate_stationavg_eventavg_statistics

from timeseries.timeseries_cfg import *
#generate_station_event_plots(configs, stations, domain_groups, time_groups)

generate_stationma_event_plots(configs, stations, domain_groups, time_groups)
'''
generate_station_event_statistics(configs, stations, domain_groups, time_groups)

generate_station_event_corr(configs, stations, domain_groups, time_groups)

generate_allstations_event_statistics(configs, stations, domain_groups, time_groups)

generate_station_eventavg_statistics(configs, stations, domain_groups, time_groups)
generate_stationavg_event_statistics(configs, stations, domain_groups, time_groups)

generate_stationavg_eventavg_statistics(configs, stations, domain_groups, time_groups)
'''