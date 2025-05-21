# test_ingest.py
"""
    The main function loads and cleans data from a CSV file using a DataIngestor class.
"""

from src.ingest import DataIngestor

def main():
    file_path = "/Users/saikirananugam/Desktop/auredia-take-home-main/data/sales_data.csv"  # Adjust if your file is differently named
    loader = DataIngestor(file_path)
    df = loader.load_and_clean()

    print("Data loaded successfully!")
    print(f"Shape: {df.shape}")
    print(df.dtypes)
    print(df.head())

if __name__ == "__main__":
    main()
