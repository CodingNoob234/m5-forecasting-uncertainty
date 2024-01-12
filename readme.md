# m5-forecasting-uncertainty
This project investigates the feature importance in the m5-forecasting-uncertainty competition from kaggle. Beneath, I've written instructions to replicate all results.

# Required Data
- Store original kaggle data in ./data/m5-forecasting-accuracy 
- Store sample submission for uncertainty in ./data/m5-forecasting-uncertainty
(create folders of non-existent)

# Required Folders to Create
- ./data/uncertainty/tables/
- ./data/uncertainty/cv_template/
- ./data/uncertainty/fold_1802/features/
- ./data/uncertainty/fold_1802/final_submissions/
- ./data/uncertainty/fold_1802/grouped/
- ./data/uncertainty/fold_1802/models/
- ./data/uncertainty/fold_1802/temp_submissions/
- similar for fold_i, i = 1830, 1858, 1886, 1914

# Required Files to Create
- ./data/uncertainty/all_results.json, containing empty dictionary {}

# Usage of Code and Replicating Results
Due to efficiency and the complexity of the project, the code is divided into multiple notebooks, structured in the following way:
- un_compute_weights_cv.ipynb: for computing the WSPL, we need to compute the weights based on historic revenue. For each of the folds, the weights are determined by the item revenue generated in the most recent month. These weights are stored in a csv file (see ./data/uncertainty/fold_{fold number}/weights_validation.csv)
- un_feature_engineering_quantiles.ipynb: loads raw competition data, computes all features for each fold and saves the features in csv file (see ./uncertainty/fold_{fold number}/features). The column names of the features contain prefixes, identifying the group of features they are in. This is used in the training phase to train models only including or excluding certain prefixes, reducing unnecesary manual work.
- un_training_model_quantiles.ipynb: loads the precomputed features, trains and evaluates the models for different quantiles and folds. We can repeat this process for different sets of features. In the final boxes of the notebooks, certain 'experiments' are defined already, containing lists of interesting feature sets to compare. This could be the comparison of moving averages, exponential weighted means or a combination for example. The models, forecasts and WSPL scores are all efficiently saved in the folders created above.
- un_visualising_results_quantiles.ipynb: presents the results in a nice and clear manner. The stored models can be loaded to plot the feature importance. We can also calculate the average WSPL scores for each of the feature combinations and create nice tables. Lastly, for each of the desired models, the predictions can be plotted, together with the actual sales.
- w_exploratory_data_analysis.ipynb: can be used to replicate all figures, mostly used to support each of the selected features. All figures will be stored under ./figures/eda/.
- un_significance_test2.ipynb: tests the statistical significance of the feature comparisons. The WSPL is computed, EXCEPT aggregation over time. I.e. the WSPL for t=1,..,28 days. For every two sets of features, the difference in WSPL is taken. The Diebold Mariano test is then performend on these residuals for t=1,...,28, for all 5 folds combined. The Diebold Mariano test results are all stored in tables under ./tables/