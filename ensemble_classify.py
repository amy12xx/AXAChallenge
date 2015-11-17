import os
import random
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split, ShuffleSplit, KFold
from sklearn.preprocessing import Binarizer
from sklearn.metrics import roc_auc_score

from train import train_ensemble

BASE_PATH = 'D:/python_files/connected_cars/data/feature_data2_smooth/'
SAVE_FILE = 'D:/python_files/connected_cars/data/submission_ensemble_featset2_smooth.csv'
np.random.seed(3564)

def create_data_set_for_driver(driver, driverslist):
	# Create dataset for driver with (+) samples of driver and (-) samples of other drivers
	# shuffle drivers list
	try:
		driverslist.remove(driver)
	except:
		pass
	random.shuffle(driverslist)

	dataset = np.array(pd.read_csv(BASE_PATH + driverslist[0]))
	np.random.shuffle(dataset)
	dataset = dataset[:10,:-1]
	dataset = np.column_stack((dataset, np.zeros(dataset.shape[0], dtype=float)))

	for d in driverslist[1:20]:
		# stack to original array
		data = np.array(pd.read_csv(BASE_PATH + d))
		np.random.shuffle(data)
		data = data[:10,:-1]
		data = np.column_stack((data, np.zeros(data.shape[0], dtype=float)))
		dataset = np.vstack((dataset, data))

	# print 'Shape after adding other driver data'
	# print dataset.shape
	return dataset

def train_to_submit(driverslist):
	# load files - had to do this cuz it takes too long to complete due to size of data. This way you can run again from where it stopped.
	# #####################################################################################
	driversdone = []
	driverdonelistfile = open('D:/python_files/connected_cars/code/driverdonelist.txt')
	for line in driverdonelistfile:
		driversdone.append(line.strip('\n'))
	print len(driversdone)

	driversremaining = list(set(driverslist) - set(driversdone))
	print 'remaining drivers'
	print len(driversremaining)
	# #####################################################################################
	
	for i, driver in enumerate(driversremaining): # Change driverslist to driversremaining if using the above code
		driver_num = int(driver.strip('Driver').strip('.csv').strip('trips'))
		driver_dataset = np.array(pd.read_csv(BASE_PATH + driver))

		print driver

		other_driver_dataset = create_data_set_for_driver(driver, driverslist)

		predictions = np.zeros(driver_dataset.shape[0])
		kf = KFold(driver_dataset.shape[0], n_folds=20)

		for train_idx, test_idx in kf:
			train = driver_dataset[train_idx]
			train = np.vstack((train, other_driver_dataset[train_idx]))
			test = driver_dataset[test_idx]
			
			pred_probas = train_ensemble(train[:,1:-1], train[:,-1], test[:,1:-1], test[:,-1])

			predictions[test_idx] = pred_probas[:,1]

		ids = driver_dataset[:,0]
		final_submission = np.column_stack((ids, predictions))

		# now save to file
		pd.DataFrame(final_submission).to_csv(SAVE_FILE, sep=',', header=None, index=False, mode='a')

def train_local(driverslist):
	# load files - had to do this cuz it takes too long to complete due to size of data. This way you can run again from where it stopped.
	# driversdone = []
	# driverdonelistfile = open('C:/python_files/connected_cars/driverdonelist.txt')
	# for line in driverdonelistfile:
	# 	driversdone.append(line.strip('\n'))
	# print len(driversdone)

	# driversremaining = list(set(driverslist) - set(driversdone))
	# print 'remaining drivers'
	# print len(driversremaining)
	# print driversremaining
	
	auc = []

	for i, driver in enumerate(driverslist): # Change driverslist to driversremaining if using the above code
		driver_num = int(driver.strip('Driver').strip('.csv').strip('trips'))
		driver_dataset = np.array(pd.read_csv(BASE_PATH + driver))

		other_driver_dataset = create_data_set_for_driver(driver, driverslist)

		predictions = np.zeros(driver_dataset.shape[0])
		kf = KFold(driver_dataset.shape[0], n_folds=20)

		driver_auc = []
		for train_idx, test_idx in kf:
			train = driver_dataset[train_idx]
			train = np.vstack((train, other_driver_dataset[train_idx]))
			test = driver_dataset[test_idx]
			test = np.vstack((test, other_driver_dataset[test_idx]))
			
			pred_probas = train_ensemble(train[:,1:-1], train[:,-1], test[:,1:-1], test[:,-1])

			y_true = Binarizer().transform(test[:,-1])
			
			score = roc_auc_score(y_true.T, pred_probas[:,1], average=None)
			driver_auc.append(score)
			auc.append(score)
			# print 'AUC score: %0.3f' % score
		print "%d Driver %d AUC score: %0.3f" % (i, driver_num, np.average(driver_auc))
	print "Overall Average AUC score: %0.3f" % np.average(auc)

def main():
	driverslist = os.listdir(BASE_PATH)
	train_to_submit(driverslist)
	# train_local(driverslist)
			
if __name__ == '__main__':
	main()
