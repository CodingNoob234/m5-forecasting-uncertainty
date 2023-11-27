# m5-forecasting-uncertainty
This project investigates the feature importance in the m5-forecasting-uncertainty competition from kaggle. Beneath, I've written instructions to replicate all results.

# Required Data
- Store original kaggle data in ./data/m5-forecasting-accuracy 
- Store sample submission for uncertainty in ./data/m5-forecasting-uncertainty

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
- ./data/uncertainty/all_results.json, containing empty {}

# Usage of Code and Replicating Results
The project eventually became quite complicated. It is therefore structured in the following way:
- un_compute_weights_cv.ipynb: for computing the WSPL, we need to compute the weights. For each of the folds, the weights are determined by the item revenue created in the most recent month. These weights are stored in a csv file (see ./data/uncertainty/fold_{fold number}/weights_validation.csv)
- un_feature_engineering_quantiles.ipynb: loads raw data, computes all features for each fold and saved in csv file (see ./uncertainty/fold_{fold number}/features). The column names in the feature csv files contain prefixes, identifying the group of features they are in. This is used in the training phase to train models only including or excluding certain prefixes, reducing unnecesary manual work.
- un_training_model_quantiles.ipynb: This notebook loads the features and trains and evaluates the models for different quantiles. We can repeat this for different sets of features and folds. In the final boxes of the notebooks, certain 'experiments' are defined already. I.e. train models for all folds and quantiles, that include 'seasonal' and 'ma', only 'seasonal', only 'ma', for example. 
- un_visualising_results_quantiles.ipynb: This notebook is used to present the results in a nice and clear manner. The stored models can be loaded to compute the feature importance. We can also calculate the average scores for each of the feature combinations and create nice tables. Lastly, for each of the desired models, the predictions can be plotted, together with the actual sales.
- w_exploratory_data_analysis.ipynb: This notebook can be used to replicate all figures, mostly used to support each of the features. All figures will be stored under ./figures/eda/.