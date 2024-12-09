
# download data from UCI ML Repository

data/raw/raw_mushroom.csv: scripts/download_data.py
	python scripts/download_data.py \
		--dataset_id=848 \
		--raw_path=data/raw/raw_mushroom.csv

# clean dataset
data/processed/cleaned_mushroom.csv \
data/processed/columns_to_drop.csv \
results/tables/missing_values_by_column.csv: \
data/raw/raw_mushroom.csv \
scripts/data_cleaning.py
	python scripts/data_cleaning.py \
		--input_path=data/raw/raw_mushroom.csv \
		--output_path=data/processed/cleaned_mushroom.csv \
		--columns_to_drop_path=data/processed/columns_to_drop.csv \
		--missing_values_path=results/tables/missing_values_by_column.csv

# validate, split into training/testing data, and preprocess 

	python scripts/split_n_preprocess.py \
		--input_path=data/processed/cleaned_mushroom.csv \
		--output_dir=data/processed \
		--results_dir=results \
		--seed=123

# produce exploratory data analysis plots

	python scripts/eda.py \
		--processed-training-data=data/processed/mushroom_train.csv \
		--plot-to=results

# train and tune models, and select the best model

	python scripts/fit_mushroom_classifier.py \
		--processed-training-data=data/processed/mushroom_train.csv \
		--preprocessor=results/models/mushroom_preprocessor.pickle \
		--pipeline-to=results/models \
		--results-to=results \
		--seed=123

# evaluate model performance on test data
	python scripts/evaluate_mushroom_predictor.py \
		--cleaned-test-data=data/processed/mushroom_test.csv \
		--pipeline-from=results/models/mushroom_best_model.pickle \
		--results-to=results \
		--seed=123
 
# render the report to a PDF

	quarto render report/*.qmd --to pdf