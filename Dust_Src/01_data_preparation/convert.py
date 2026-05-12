import pandas as pd

def convert_parquet_to_csv(parquet_file_path, csv_file_path, index=False):
    """
    Converts a Parquet file to a CSV file.

    Args:
        parquet_file_path (str): The path to the input Parquet file.
        csv_file_path (str): The path where the output CSV file will be saved.
        index (bool, optional): Whether to write the DataFrame index to the CSV file.
                                Defaults to False.
    """
    try:
        # Read the Parquet file into a pandas DataFrame
        df = pd.read_parquet(parquet_file_path)

        # Write the DataFrame to a CSV file
        df.to_csv(csv_file_path, index=index)

        print(f"Successfully converted '{parquet_file_path}' to '{csv_file_path}'")
    except FileNotFoundError:
        print(f"Error: Parquet file not found at '{parquet_file_path}'")
    except Exception as e:
        print(f"An error occurred during conversion: {e}")

# Example usage:
input_parquet_file = "GRID3km_DAILY_20x20_modelready_labeled_featured.parquet"  # Replace with your Parquet file name
output_csv_file = "GRID3km_DAILY_20x20_modelready_labeled_featured.csv"      # Replace with your desired CSV file name

convert_parquet_to_csv(input_parquet_file, output_csv_file)