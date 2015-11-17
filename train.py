import os
import sys

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split

from ensemble import EnsembleClassifier

BASE_PATH = 'D:/python_files/connected_cars/data/feature_data2/'

def train_ensemble(X, y, X_test, y_test):
	clf1 = LogisticRegression(random_state=123, solver='lbfgs')
	clf2 = RandomForestClassifier(random_state=123, n_estimators=400, criterion='gini', max_depth=15,  n_jobs=1, min_samples_leaf=5,verbose=0)
	clf5 = svm.SVC(kernel='rbf', C=0.01, class_weight='auto', probability=True)
	clf8 = GradientBoostingClassifier(n_estimators=400, min_samples_leaf=3, learning_rate=0.01, max_depth=5)
	
	clfs = [clf1, clf2, clf5, clf8]
	weights = [1,2,3,5]

	X = Normalizer().fit_transform(X)
	X_test = Normalizer().transform(X_test)

	eclf = EnsembleClassifier(clfs=clfs, voting='soft', weights=weights) 
	preds = eclf.fit(X, y).predict_proba(X_test)

	del clf1, clf2, clf5, clf8, eclf
	
	return preds

def main():
	# load files
	driverslist = os.listdir(BASE_PATH)
	driver = sys.argv[1]
	dataset = np.array(pd.read_csv(BASE_PATH + driver))
	dataset2 = np.array(pd.read_csv(BASE_PATH + 'Driver1013trips.csv'))

	dataset = np.vstack((dataset, dataset2))

	X_train, X_test, y_train, y_test = train_test_split(dataset[:,1:-1], dataset[:,-1], test_size=0.2, random_state=42)

	preds = train_ensemble(X_train, y_train, X_test, y_test)
	print preds
	
if __name__ == '__main__':
	main()
