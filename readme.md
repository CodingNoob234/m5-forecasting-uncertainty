# m5-forecasting-uncertainty
In this project the data from the m5-frecasting-uncertainty competition on kaggle is used to make a submission in the competition.


# Usage of Code and Replicating Results
- un_compute_weights_cv.ipynb: computes the evaluation weights for each fold for cross validation and stores weights in csv file (see ./uncertainty/fold_{fold number}/weights_validation.csv)
- un_feature_engineering_quantiles.ipynb: loads raw data, computes all features for each fold and saved in csv file (see ./uncertainty/fold_{fold number}/features). The column names in the feature csv files contain prefixes, identifying the group of features they are in. This is used in the training phase to train models only including or excluding certain prefixes, reducing unnecesary manual work.
- un_training_model_quantiles.ipynb: for a given set of prefixes to include or exlucde from the features, a model is trained and evaluated for each fold