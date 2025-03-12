import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


rdw_df = pd.read_csv(
    "./Data.csv",
    low_memory=False,
    nrows=5000000,
    dtype={
        "Datum eerste toelating": str,
        "Catalogusprijs": float,
        "Aantal cilinders": float,
        "Massa ledig voertuig": float,
        "Maximale constructiesnelheid": float,
    },
)

print("CSV READ")

# Convert the date column to datetime
rdw_df["Datum eerste toelating"] = pd.to_datetime(
    rdw_df["Datum eerste toelating"], errors="coerce"
)


selected_columns = [
    "Kenteken",
    "Voertuigsoort",
    "Merk",
    "Aantal cilinders",
    "Massa ledig voertuig",
    "Datum eerste toelating",
    "Maximale constructiesnelheid",
    "Catalogusprijs",
]

features_label_df = rdw_df[selected_columns].copy()
features_label_df.columns = [
    "License_Plate",
    "Vehicle_Type",
    "Make",
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
    ).astype("float64")


# 4. Extract Year, Month, Day from Registration Date
features_label_df["Reg_Year"] = features_label_df[
    "First_Registration_Date"
].dt.year.astype("float64")
features_label_df["Reg_Month"] = features_label_df[
    "First_Registration_Date"
].dt.month.astype("float64")
features_label_df["Reg_Day"] = features_label_df[
    "First_Registration_Date"
].dt.day.astype("float64")

features_label_df.drop(columns=["First_Registration_Date"], inplace=True)

numeric_columns.extend(["Reg_Year", "Reg_Month", "Reg_Day"])

features_label_df.dropna(inplace=True)

cat_cols = ["Vehicle_Type", "Make"]
for cat_col in cat_cols:
    encoder = LabelEncoder()
    # Convert to string in case they aren't strictly str
    features_label_df[cat_col] = encoder.fit_transform(
        features_label_df[cat_col].astype(str)
    )

if "License_Plate" in features_label_df.columns:
    features_label_df.drop(columns=["License_Plate"], inplace=True)

scaler = StandardScaler()
features_label_df[numeric_columns] = scaler.fit_transform(
    features_label_df[numeric_columns]
)

# Reorder columns, puts price at the end
all_cols = [col for col in features_label_df.columns if col != "Price"]
all_cols.append("Price")
features_label_df = features_label_df[all_cols]

print(
    "Number of rows in final dataset after dropping missing values:",
    len(features_label_df),
)
print("Number of columns:", features_label_df.shape[1])

features_label_df.to_csv("Price_prediction_data.csv", index=False)
print("Preprocessed data saved to 'Price_prediction_data.csv'.")
