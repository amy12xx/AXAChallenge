# AXAChallenge
Solution to AXA Kaggle challenge to determine driver pattern from GPS data

1. Driver iterates over all drivers in the dataset to generate features using DrivingTrip.py
2. ensemble_classify.py performs classification for each driver using % of driver data + non-driver data as training set, and % of driver_data as test set. Cross-validation is done to make sure we have predictions for all driver data, to submit to Kaggle.
3. train.py implements the ensemble, using Sebastian Raschka's ensemble classifier.

Data must be retrieved from Kaggle website, and appropriate path locations changed in the code.
