import pandas as pd
import os

def combine_datasets(input_dir="backend/datasets", output_file="combined_dataset.csv"):
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read and combine all CSV files in the input directory
    dataframes = [pd.read_csv(f"{input_dir}/{file}") for file in os.listdir(input_dir) if file.endswith(".csv")]
    
    # Concatenate all dataframes into one dataset
    dataset = pd.concat(dataframes, ignore_index=True)
    
    # Save the combined dataset to the specified output file
    dataset.to_csv(output_file, index=False)
    print(f"Combined dataset saved to {output_file}.")

# Run the function
combine_datasets(input_dir="backend/datasets", output_file="backend/datasets/combined_dataset.csv")
