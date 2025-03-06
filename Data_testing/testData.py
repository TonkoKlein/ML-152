import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import StringIO


# Increase visibility of potential issues
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

# path = "../Data.csv"
apiEndpoint = "https://opendata.rdw.nl/resource/m9d7-ebf2.csv?$limit=100000"
response = requests.get(apiEndpoint)
rdw_df = pd.read_csv(
    StringIO(response.text),
    low_memory=False,
    dtype={
        "datum_eerste_toelating": str,
        "catalogusprijs": float,
        "aantal_cilinders": float,
        "massa_ledig_voertuig": float,
        "maximale_constructiesnelheid": float,
    },
)


rdw_df["datum_eerste_toelating"] = pd.to_datetime(
    rdw_df["datum_eerste_toelating"], errors="coerce"
)

selected_columns = [
    "kenteken",
    "voertuigsoort",
    "aantal_cilinders",
    "massa_ledig_voertuig",
    "datum_eerste_toelating",
    "maximale_constructiesnelheid",
    "catalogusprijs",
]

features_label_df = rdw_df[selected_columns].copy()

features_label_df.columns = [
    "License_Plate",
    "Vehicle_Type",
    "Number_of_Cylinders",
    "Vehicle_Mass",
    "First_Registration_Date",
    "Max_Speed",
    "Price",
]

numeric_columns = ["Number_of_Cylinders", "Vehicle_Mass", "Max_Speed", "Price"]
for col in numeric_columns:
    features_label_df[col] = pd.to_numeric(
        features_label_df[col], errors="coerce"
    ).astype("Int64")

# Remove rows with missing price, ensuring at least some data remains
features_label_df = features_label_df.dropna(subset=["Price"])

# Replace missing cylinder counts with the most common value
most_common_cylinders = features_label_df["Number_of_Cylinders"].mode()[0]
features_label_df["Number_of_Cylinders"] = features_label_df[
    "Number_of_Cylinders"
].fillna(most_common_cylinders)

# Save to CSV
features_label_df.to_csv("Price_prediction_data.csv", index=False)
