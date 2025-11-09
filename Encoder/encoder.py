import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split

def detect_categorical_columns(df):

    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return cat_cols

def label_encode(df, columns):
    le = LabelEncoder()
    for col in columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def one_hot_encode(df, columns):
    df = pd.get_dummies(df, columns=columns, drop_first=True)
    return df

def target_encode(df, columns, target):
    te = TargetEncoder(cols=columns)
    df[columns] = te.fit_transform(df[columns], df[target])
    return df

#def main():
    print("Encoder Demo Started")

    data = pd.read_csv(r"Input/encodertrain.csv")
    df = pd.DataFrame(data)
    print("\n Original Data:")
    print(df)

    cat_cols = detect_categorical_columns(df)
    print(f"\nDetected Categorical Columns: {cat_cols}")

    df_label = label_encode(df.copy(), cat_cols)
    print("\n Label Encoded Data:")
    print(df_label)

    # One-Hot Encoding
    df_onehot = one_hot_encode(df.copy(), cat_cols)
    print("\n One-Hot Encoded Data:")
    print(df_onehot)


    df_target = target_encode(df.copy(), cat_cols, target=cat_cols[0])
    print("\n Target Encoded Data:")
    print(df_target)

    df_label.to_csv(r"Encoder/Output/encoded_label.csv", index=False)
    df_onehot.to_csv(r"Encoder/Output/encoded_onehot.csv", index=False)
    df_target.to_csv(r"Encoder/Output/encoded_target.csv", index=False)
    
    print("\nEncoded datasets saved successfully!")
