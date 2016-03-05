'''
Created on: Sept 8, 2015

@author: Amanda_Dsouza@Infosys.com

input array header: [max_speed,avg_speed,std_dev_speed,avg_acceleration,avg_sudden_acceleration,
high_acceleration_time,high_deceleration_time,avg_positive_acceleration,std_dev_positive_acceleration,
avg_deceleration,deceleration_std_dev,
              low_speed_time,idle_time_percent,stop_time_by_traversed_distance,
              max_slip_angle,avg_slip_angle,slip_angle_std_dev,
              speed_for_low_time_interval,speed_for_mid_time_interval,speed_for_high_time_interval,
              speed_over_ftp_city_limit,slip_over_interval_by_total_dist,slip_over_interval_by_total_time,
              avg_slip_angle_5s, max_slip_angle_5s, max_speed_5s, avg_speed_5s, sudden_acceleration_5s]
'''
from math import sqrt
import os
import re
from time import time
import subprocess
from subprocess import PIPE

from pyspark.mllib.clustering import KMeans
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import numpy as np
import pandas as pd

BASE_PATH = "/usr/local/spark/car/"

def exec_command(command):
    """
    Execute the command and return the exit status.
    """
    pobj = subprocess.Popen(command, stdout=PIPE, stderr=PIPE, shell=True)

    stdo, stde = pobj.communicate()
    exit_code = pobj.returncode

    return exit_code, stdo, stde

def read_file_path(path):
	command = "hadoop fs -ls %s" % path
	exit_code, stdo, stde = exec_command(command)

	flist = []
	lines = stdo.split('\n')
	for line in lines:
		try:
			ls = line.split()
			flist.append(ls[len(ls)-1])
			if len(ls) == 8:
				# this is a file description line
				fname=ls[-1]
				flist.append(fname)
		except:
			continue
	print 'Number of items found: %d' % len(flist)
	return set(flist)

def error(point, clusters):
	center = clusters.centers[clusters.predict(point)]
	return sqrt(sum([x**2 for x in (point - center)]))

def get_cluster_id(clusters, point):
	point_center = clusters.centers[clusters.predict(point)]
	centers = clusters.centers
	cluster_id = -1

	for i, center in enumerate(centers):
		if np.array_equal(np.array(point_center),np.array(centers[i])):
			cluster_id = i
			break

	return cluster_id

def cluster_data(sc, qc):
	drivers = read_file_path(BASE_PATH)
	print "Number of drivers: %d" % len(drivers)

	# Load and parse the data
	for i, dr in enumerate(drivers):
		# extract driver number from path
		dr_num = re.search("[0-9]+$", dr.strip())

		if dr_num:
			dr_num = dr_num.group(0)
			if dr_num == '1018':
				continue
		else:
			print 'driver number error for %s' % dr 
			continue

		dr_data = sc.textFile("hdfs://" + dr + "/" + dr_num + "_all_trips.txt")

		data = dr_data.map(lambda row: [float(x) for x in row.split(',')])

		if i == 0:
			all_data = data
		else:
			all_data = all_data.union(data)

		data.unpersist()

	print 'Total number of records: %d' % all_data.count()

	# Build the model (cluster the data), k = Number of clusters
	k = 5 
	t = time()
	clusters = KMeans.train(all_data, k, maxIterations=100, runs=100, initializationMode="random", )
	print 'KMeans took %.2f seconds' % (time() - t)

	# Compute cost
	WSSSE_map = all_data.map(lambda point: error(point, clusters))

	# Join cluster ID to original data
	all_data_w_cluster = all_data.map(lambda point: np.hstack((point, get_cluster_id(clusters, point))))

	# all_data_w_cluster.saveAsTextFile("hdfs:///usr/local/spark/kmeans/results.txt")

	for i in xrange(0,k):
		subset = all_data_w_cluster.filter(lambda x: x[-1] == i)
		print "Number of items in cluster %d: %d" % (i, subset.count())
		# Computer functions on different features:
		all_features_average = subset.sum() / subset.count()
		print 'Average of all features'
		print all_features_average
	
	WSSSE = all_data.map(lambda point: error(point, clusters)).reduce(lambda x, y: x + y)
	print("Within set sum of squared error: " + str(WSSSE))

if __name__ == '__main__':
    appName="KMeans on Connected Cars"
    master = "spark://VICPINVPZ09:7077"

    conf = SparkConf().setAppName(appName).setMaster(master)
    sc = SparkContext(conf=conf)
    qc = SQLContext(sc)

    cluster_data(sc, qc)