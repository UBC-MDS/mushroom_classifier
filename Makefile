.PHONY: all clean-data clean-results clean-models clean-all



# download data
data/raw/raw_mushroom.csv: scripts/download_data.py
	python scripts/download_data.py \
    	--dataset_id=848 \
    	--raw_path=data/raw/raw_mushroom.csv

# clean data by removing some columns and dropping missing values,
# log removed columns and missing entries in columns_to_drop.csv and missing_values_by_column.csv
data/processed/cleaned_mushroom.csv \
data/processed/columns_to_drop.csv \
results/tables/missing_values_by_column.csv: scripts/data_cleaning.py data/raw/raw_mushroom.csv
	python scripts/data_cleaning.py \
 		--input_path=data/raw/raw_mushroom.csv \
    	--output_path=data/processed/cleaned_mushroom.csv \
    	--columns_to_drop_path=data/processed/columns_to_drop.csv \
    	--missing_values_path=results/tables/missing_values_by_column.csv

# Validate and split data into train and test sets, and output split data and correlation matrix for features.
# Produce training target variables distribution, build and save the preprocessor, and store the
# preprocessed training/testing data.
results/tables/schema_validate.csv \
data/processed/mushroom_train.csv \
data/processed/mushroom_test.csv \
results/tables/numeric_correlation_matrix.csv \
results/figures/target_variable_distribution.png \
results/models/mushroom_preprocessor.pickle \
data/processed/scaled_mushroom_train.csv \
data/processed/scaled_mushroom_test.csv : scripts/split_n_preprocess.py data/processed/cleaned_mushroom.csv
	python scripts/split_n_preprocess.py \
    	--input_path=data/processed/cleaned_mushroom.csv \
    	--output_dir=data/processed \
    	--results_dir=results \
    	--seed=123

# Perform exploratory data analysis on training dataset, by producing histograms of numeric features
# by target and frequency tables of categorical data
results/tables/*_summary.csv results/figures/*_histogram.png: scripts/eda.py data/processed/mushroom_train.csv
	python scripts/eda.py \
    	--processed-training-data=data/processed/mushroom_train.csv \
    	--plot-to=results/figures \
    	--table-to=results/tables

# Build three model pipelines with the preprocessor, using KNN, SVM, and Logistic Regression classifiers. 
# Tune each model using cross validation on the training data, and select the model with the best F2 score.
# Record cross-validation results and save the selected best model. Also save the confusion matrix for the best model.

results/tables/cross_val_results.csv \
results/models/mushroom_best_model.pickle \
results/figures/train_confusion_matrix.png: scripts/fit_mushroom_classifier.py data/processed/mushroom_train.csv
	python scripts/fit_mushroom_classifier.py \
    	--processed-training-data=data/processed/mushroom_train.csv \
    	--preprocessor=results/models/mushroom_preprocessor.pickle \
   		--pipeline-to=results/models \
    	--results-to=results \
    	--seed=123

# Evaluate the model performance on the test data, output final scores and test confusion matrix
results/tables/test_confusion_matrix.csv \
results/tables/test_scores.csv: scripts/evaluate_mushroom_predictor.py data/processed/mushroom_test.csv results/models/mushroom_best_model.pickle
	python scripts/evaluate_mushroom_predictor.py \
    	--cleaned-test-data=data/processed/mushroom_test.csv \
    	--pipeline-from=results/models/mushroom_best_model.pickle \
    	--results-to=results \
    	--seed=123

# Render report to HTML and PDF
report/mushroom_classifier_report.html report/mushroom_classifier_report.pdf: report/mushroom_classifier_report.qmd \
report/references.bib \
data/processed/cleaned_mushroom.csv \
data/processed/columns_to_drop.csv \
results/tables/numeric_correlation_matrix.csv \
results/figures/target_variable_distribution.png \
results/tables/*_summary.csv \
results/figures/*_histogram.png \
results/tables/cross_val_results.csv \
results/figures/train_confusion_matrix.png \
results/tables/test_confusion_matrix.csv \
results/tables/test_scores.csv
	quarto render report/mushroom_classifier_report.qmd --to pdf
	quarto render report/mushroom_classifier_report.qmd --to html

# run entire analysis
all: report/mushroom_classifier_report.html report/mushroom_classifier_report.pdf

# remove raw and preprocessed data
clean-data:
	rm -r data/raw/*
	rm -r data/processed/*

# remove all tables,figures, etc.
clean-results:
	rm -r results/figures/*
	rm -r results/tables/*

# remove model and preprocessor
clean-models:
	rm -r results/models/*

# remove rendered reports
clean-reports:
	rm -rf report/mushroom_classifier_report.html \
			report/mushroom_classifier_report.pdf \
			report/mushroom_classifier_report_files

clean-all: clean_data clean-results clean-models clean-reports
	
