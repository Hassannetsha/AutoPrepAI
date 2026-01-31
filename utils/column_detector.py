def detect_categorical_columns(df):
    return df.select_dtypes(include=['object', 'category']).columns.tolist()
