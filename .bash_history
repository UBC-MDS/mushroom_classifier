cd ..
python scripts/fit_mushroom_classifier.py     --processed-training-data=data/processed/mushroom_train.csv     --preprocessor=results/models/mushroom_preprocessor.pickle     --pipeline-to=results/models     --results-to=results     --seed=123
python scripts/evaluate_mushroom_predictor.py     --cleaned-test-data=data/processed/mushroom_test.csv     --pipeline-from=results/models/mushroom_best_model.pickle     --results-to=results     --seed=123
python scripts/evaluate_mushroom_predictor.py     --cleaned-test-data=data/processed/mushroom_test.csv     --pipeline-from=results/models/mushroom_best_model.pickle     --results-to=results     --seed=123
