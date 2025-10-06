import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Employee Feature Selection", layout="wide")
st.title("RandomForest Feature Selection with Uploaded Data")

# =========================
# Upload CSV File
# =========================
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Ask user which column is the target
    target_column = st.selectbox("Select target column", df.columns)
    y = df[target_column]
    
    # Features (drop target)
    X = df.drop(columns=[target_column])
    
    # Convert categorical columns to numeric (one-hot)
    X = pd.get_dummies(X)
    st.subheader("Features after encoding")
    st.write(X.columns.tolist())
    
    # =========================
    # Train/Test Split
    # =========================
    test_size = st.slider("Test size (fraction)", 0.1, 0.5, 0.3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # =========================
    # Feature Selection
    # =========================
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    threshold_option = st.selectbox("Threshold for feature selection", ["median", "mean"])
    selector = SelectFromModel(rf, threshold=threshold_option)
    selector.fit(X_train, y_train)
    
    selected_features = X.columns[selector.get_support()]
    st.subheader("Selected Features")
    st.write(selected_features.tolist())
    
    X_train_sel = selector.transform(X_train)
    X_test_sel = selector.transform(X_test)
    
    # =========================
    # Train Model on Selected Features
    # =========================
    rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_selected.fit(X_train_sel, y_train)
    y_pred = rf_selected.predict(X_test_sel)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.subheader("Model Accuracy on Selected Features")
    st.write(f"{accuracy*100:.2f}%")
else:
    st.info("Please upload a CSV file to start the analysis.")
