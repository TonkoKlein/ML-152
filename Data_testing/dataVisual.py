import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🔹 Load the cleaned CSV file
file_path = "Price_prediction_data.csv"  # Update path if needed
features_label_df = pd.read_csv(file_path)

# 🔹 Convert date column if present
if "First_Registration_Date" in features_label_df.columns:
    features_label_df["First_Registration_Date"] = pd.to_datetime(
        features_label_df["First_Registration_Date"], errors="coerce"
    )

# 🔹 Basic info check
print(features_label_df.info())
print(features_label_df.describe())

# 🔹 Histogram of Vehicle Prices
plt.figure(figsize=(10, 5))
sns.histplot(features_label_df["Price"], bins=50, kde=True)
plt.xlabel("Price (€)")
plt.ylabel("Number of Vehicles")
plt.title("Distribution of Vehicle Prices")
plt.show()

# 🔹 Count of Vehicle Types
if "Vehicle_Type" in features_label_df.columns:
    plt.figure(figsize=(12, 5))
    sns.countplot(
        y=features_label_df["Vehicle_Type"],
        order=features_label_df["Vehicle_Type"].value_counts().index,
    )
    plt.xlabel("Number of Vehicles")
    plt.ylabel("Vehicle Type")
    plt.title("Count of Vehicles by Type")
    plt.show()

# 🔹 Boxplot: Price vs. Number of Cylinders
if "Number_of_Cylinders" in features_label_df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x=features_label_df["Number_of_Cylinders"], y=features_label_df["Price"]
    )
    plt.xlabel("Number of Cylinders")
    plt.ylabel("Price (€)")
    plt.title("Vehicle Price vs. Number of Cylinders")
    plt.show()

# 🔹 Scatterplot: Vehicle Mass vs. Price
if "Vehicle_Mass" in features_label_df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=features_label_df["Vehicle_Mass"], y=features_label_df["Price"], alpha=0.5
    )
    plt.xlabel("Vehicle Mass (kg)")
    plt.ylabel("Price (€)")
    plt.title("Vehicle Mass vs. Price")
    plt.show()

# 🔹 Line Plot: Price Trends Over Time
if "First_Registration_Date" in features_label_df.columns:
    features_label_df["Registration_Year"] = features_label_df[
        "First_Registration_Date"
    ].dt.year

    plt.figure(figsize=(12, 6))
    sns.lineplot(x=features_label_df["Registration_Year"], y=features_label_df["Price"])
    plt.xlabel("Year of Registration")
    plt.ylabel("Average Price (€)")
    plt.title("Price Trends Over Time")
    plt.show()

# 🔹 Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(features_label_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
