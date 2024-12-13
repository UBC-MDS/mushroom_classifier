import os
import pandas as pd

def summarize_categorical_features(data, output_dir):
    """
    Summarize categorical features in the dataset and save tables as CSV.
    
    Parameters:
        data (pd.DataFrame): The processed training data containing features and class labels.
        output_dir (str): Directory to save the summary tables.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    categorical_features = [col for col in data.columns if data[col].dtype == "object"]
    
    for feature in categorical_features:
        summary = data[feature].value_counts(normalize=False).to_frame(name="Count")
        summary["Proportion"] = data[feature].value_counts(normalize=True)
        
        output_path = os.path.join(output_dir, f"{feature}_summary.csv")
        summary.to_csv(output_path)
        print(f"Saved summary table for '{feature}' at: {output_path}")