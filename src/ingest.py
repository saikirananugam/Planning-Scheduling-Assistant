# The `DataIngestor` class loads and cleans a sales dataset from a CSV file, performs feature
# engineering including date parsing and outlier detection using the IQR method, and returns a cleaned
# and enhanced DataFrame.

import pandas as pd
import numpy as np

class DataIngestor:
    def __init__(self, filepath: str):
        """
        Initialize the DataIngestor with the path to the CSV file.
        :param filepath: str - path to the sales data CSV file
        """
        self.filepath = filepath

    def load_and_clean(self):
        """
        Load the CSV and perform basic cleaning (date parsing, null handling, etc.)

        Feature engineering includes:
        - Parsing date columns
        - Calculating delay days and delay flag
        - Computing lead time and pickup lead
        - Extracting temporal features like month, year, week
        - Creating an urgency flag
        - Deriving total units from Line Amount and SRP1
        - Removing fully duplicated rows
        - Logging outliers using IQR method for selected numeric columns

        Returns:
            pd.DataFrame: Cleaned and feature-enhanced sales dataset
        """
        # Load the raw data from CSV
        df = pd.read_csv(self.filepath)

        # Drop fully duplicated rows
        initial_rows = df.shape[0]
        df = df.drop_duplicates()
        final_rows = df.shape[0]
        if initial_rows != final_rows:
            print(f"🧹 Removed {initial_rows - final_rows} duplicate rows.")

        # List of date columns to convert to datetime format
        date_cols = ['ECD', 'Line Creation', 'Schedule Pick Date', 'Promised Delivery Date']
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    # The code snippet you provided is part of the feature engineering process in the `load_and_clean`
    # method of the `DataIngestor` class. Here's a breakdown of what each part of the code is doing:

        # Feature Engineering
        df['delay_days'] = (df['ECD'] - df['Promised Delivery Date']).dt.days
        df['delay_flag'] = (df['delay_days'] > 0).astype(int)
        df['lead_time'] = (df['ECD'] - df['Line Creation']).dt.days
        df['pickup_lead'] = (df['Schedule Pick Date'] - df['Line Creation']).dt.days
        df['month'] = df['Promised Delivery Date'].dt.month
        df['year'] = df['Promised Delivery Date'].dt.year
        df['week'] = df['Promised Delivery Date'].dt.isocalendar().week
        df['is_urgent'] = df['TL SO Alert'].notna().astype(int)
        df['total_units'] = df['Line Amount'] / df['SRP1']

        # Check for any missing values
        if df.isnull().sum().any():
            print("🔍 Warning: Null values detected in the dataset:")
            print(df.isnull().sum()[df.isnull().sum() > 0])

       # The code snippet you provided is performing outlier detection using the Interquartile Range
       # (IQR) method for specific numeric columns in the DataFrame. Here's a breakdown of what each
       # part of the code is doing:
        # Outlier detection using IQR method
        outlier_cols = ['lead_time', 'total_units', 'pickup_lead', 'delay_days', 'Line Amount']
        for col in outlier_cols:
            if col in df.columns:
                Q1 = np.percentile(df[col].dropna(), 25)
                Q3 = np.percentile(df[col].dropna(), 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_count = outlier_mask.sum()
                if outlier_count > 0:
                    print(f"{outlier_count} outliers detected in '{col}'")

        return df
