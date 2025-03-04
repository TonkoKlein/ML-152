
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Increase visibility of potential issues
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

rdw_df = pd.read_csv('Open_Data_RDW__Gekentekende_voertuigen_20250124.csv', 
                     low_memory=False, 
                     dtype={
                         'Datum eerste toelating': str,
                         'Catalogusprijs': float,
                         'Aantal cilinders': float,
                         'Massa ledig voertuig': float,
                         'Maximale constructiesnelheid': float
                     })


rdw_df['Datum eerste toelating'] = pd.to_datetime(rdw_df['Datum eerste toelating'], errors='coerce')

selected_columns = [
    'Kenteken',
    'Voertuigsoort',
    'Aantal cilinders',
    'Massa ledig voertuig',
    'Datum eerste toelating',
    'Maximale constructiesnelheid',
    'Catalogusprijs'
]

features_label_df = rdw_df[selected_columns].copy()

features_label_df.columns = [
    'License_Plate', 
    'Vehicle_Type', 
    'Number_of_Cylinders', 
    'Vehicle_Mass', 
    'First_Registration_Date', 
    'Max_Speed', 
    'Price'
]

numeric_columns = ['Number_of_Cylinders', 'Vehicle_Mass', 'Max_Speed', 'Price']
for col in numeric_columns:
    features_label_df[col] = pd.to_numeric(features_label_df[col], errors='coerce')

# Remove rows with missing price, ensuring at least some data remains
features_label_df = features_label_df.dropna(subset=['Price'])

# Save to CSV
features_label_df.to_csv('Price_prediction_data.csv', index=False)
