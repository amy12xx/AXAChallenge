'''
Driver class is used to generate the features for the driver.
Main method iterates over all drivers to generate the features for all drivers.
'''
import os
import codecs

from joblib import Parallel, delayed
import pandas as pd

import DrivingTrip

SAVE_PATH = 'D:/python_files/connected_cars/data/feature_data2_smooth/'

class Driver:
	def __init__(self, id, path):
		self.driver_id = id
		self.driving_trips = []

		# Get all driving trips of driver
		trips = os.listdir(path)
		for trip in trips:
			if '.csv' in trip:
				full_path = path + "/" + trip
				trip_num = trip.strip('.csv')
				X = pd.read_csv(full_path)
				dt = DrivingTrip.DrivingTrip(X, trip_num)
				dt.fit()
				self.driving_trips.append(dt)

	def __iter__(self):
		return self

def trip_string(driver_id, dt):
	row = [
		str(str(driver_id) + "_" + str(dt.trip_num)), \
		str(dt.max_speed), \
		str(dt.avg_speed), \
		str(dt.std_dev_speed), \
		str(dt.avg_acceleration), \
		str(dt.avg_sudden_acceleration), \
		str(dt.high_acceleration_time), \
		str(dt.high_deceleration_time), \
		str(dt.avg_positive_acceleration), \
		str(dt.std_dev_positive_acceleration), \
		str(dt.avg_deceleration), \
		str(dt.deceleration_std_dev), \
		str(dt.low_speed_time), \
		str(dt.idle_time_percent), \
		str(dt.stop_time_by_traversed_distance), \
		str(dt.max_slip_angle), \
		str(dt.avg_slip_angle), \
		str(dt.slip_angle_std_dev), \
		str(dt.speed_for_low_time_interval), \
		str(dt.speed_for_mid_time_interval), \
		str(dt.speed_for_high_time_interval), \
		str(dt.speed_over_ftp_city_limit), \
		str(dt.slip_over_interval_by_total_dist), \
		str(dt.slip_over_interval_by_total_time), \
		str(dt.avg_slip_angle_5s), \
		str(dt.max_slip_angle_5s), \
		str(dt.max_speed_5s), \
		str(dt.avg_speed_5s), \
		str(dt.sudden_acceleration_5s), \
		str(driver_id)
		]

	trip_string = ', '.join(row)
	return trip_string

def main():
    path = 'D:/python_files/connected_cars/data/smooth/drivers'
    drivers = os.listdir(path)
    for driver in drivers:
    	driver_path = path + '/' + str(driver)
    	print driver_path
    	dr = Driver(driver, driver_path)
		# Iterate through trips for driver
    	with codecs.open(SAVE_PATH + "Driver" + str(driver) + "trips.csv", 'w', encoding='utf-8') as trip_file:
    		header_row = ['id', \
    		'max_speed', \
    		'avg_speed', \
    		'std_dev_speed', \
    		'avg_acceleration', \
    		'avg_sudden_acceleration', \
    		'high_acceleration_time', \
    		'high_deceleration_time', \
    		'avg_positive_acceleration', \
    		'std_dev_positive_acceleration', \
    		'avg deceleration', \
    		'std dev deceleration', \
    		'low_speed_time', \
    		'idle_time_percent', \
    		'stop_time_by_traversed_distance', \
    		'max_slip_angle', \
    		'avg_slip_angle', \
    		'std_dev_slip', \
    		'speed_for_low_time_interval', \
    		'speed_for_mid_time_interval', \
    		'speed_for_high_time_interval', \
    		'speed_over_ftp_city_limit', \
    		'slip_over_interval_by_total_dist', \
    		'slip_over_interval_by_total_time', \
    		'avg_slip_angle_5s', \
    		'max_slip_angle_5s', \
    		'max_speed_5s', \
    		'avg_speed_5s', \
    		'sudden_acceleration_5s', \
    		'driver_id']
    		
    		trip_file.write(", ".join(header_row) + '\n')
	    	for i, dt in enumerate(dr.driving_trips):
    			ts = trip_string(dr.driver_id, dt)
    			trip_file.write(ts + '\n')

if __name__ == '__main__':
	main()
