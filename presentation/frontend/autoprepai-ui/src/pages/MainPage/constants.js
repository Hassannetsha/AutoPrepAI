export const ACTION_TO_INTENT = {
  "Handle Missing Values": "handle_missing_values",
  "Remove Outliers": "remove_outliers",
  "Remove Duplicates": "remove_duplicates",
  "Detect Feature Inconsistency": "remove_inconsistencies",
  "Scale Data": "scale_numerical",
  "Encode Data": "encode_categorical",
  "Feature Engineering": "feature_engineering",
  "Data Standardization": "standardize_data",
};

export const ACTIONS = [
  "Handle Missing Values",
  "Remove Outliers",
  "Remove Duplicates",
  "Detect Feature Inconsistency",
  "Scale Data",
  "Encode Data",
  "Feature Engineering",
  "Data Standardization",
];

export const ALLOWED_MIME_TYPES = [
  "text/csv",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  "application/vnd.ms-excel",
];
