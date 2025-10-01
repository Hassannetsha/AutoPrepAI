import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt # Optional, for visualization
import glob

csv_files = glob.glob("./Input/placement.csv", recursive=True)
if not csv_files:
    raise FileNotFoundError("Please provide a dataset in ./Input/placement.csv")

DATA_PATH = csv_files[0]
print("Using dataset:", DATA_PATH)

dataframe = pd.read_csv(DATA_PATH)
print("\nOriginal Dataset Shape:", dataframe.shape)
print(dataframe)

numerical_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
if not numerical_cols:
    raise Exception("No numerical columns found for outlier detection!")

print("\nNumerical columns considered for outliers:", numerical_cols)
print(dataframe[numerical_cols].describe())

print("\nRunning Isolation Forest...")

iso = IsolationForest(
    contamination="auto",
    random_state=42,
    n_estimators=200,      
    max_samples="auto",    
    max_features=1.0       
)

X = dataframe[numerical_cols].fillna(0)  
iso.fit(X)

dataframe['IF_score'] = iso.decision_function(X)
dataframe['is_outlier_IF'] = iso.predict(X)  

print("\nIsolationForest predictions (+1=inlier, -1=outlier):")
print(dataframe['is_outlier_IF'].value_counts())

dataframe_IF_cleaned = dataframe[dataframe['is_outlier_IF'] == 1]\
                        .drop(columns=['is_outlier_IF'])
print(f"\nAfter IsolationForest filtering: {dataframe_IF_cleaned.shape[0]} rows remain "
      f"(from {dataframe.shape[0]} original)")

# Optional, for visualization
if len(numerical_cols) >= 2:
    plt.figure(figsize=(6,4))
    plt.scatter(dataframe[numerical_cols[0]], dataframe[numerical_cols[1]],
                c=dataframe['is_outlier_IF'], cmap='coolwarm', edgecolor='k')
    plt.xlabel(numerical_cols[0])
    plt.ylabel(numerical_cols[1])
    plt.title("IsolationForest Outlier Detection")
    plt.show()

plt.figure(figsize=(6,4))
plt.hist(dataframe["IF_score"], bins=30, edgecolor="k")
plt.title("Distribution of Isolation Forest Scores")
plt.xlabel("IF_score")
plt.ylabel("Frequency")
plt.show()

# Optional, for visualization
print(dataframe_IF_cleaned)
print("\nSummary of Outlier Handling:")
print(f"- Original dataset rows: {dataframe.shape[0]}")
print(f"- After IsolationForest: {dataframe_IF_cleaned.shape[0]} rows remain")
print(f"- Outliers detected: {(dataframe.shape[0] - dataframe_IF_cleaned.shape[0])}")
