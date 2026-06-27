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

export const INITIAL_BOT_MESSAGE = {
  sender: "bot",
  text: "Hello! I'm your AutoPrepAI assistant. Upload a dataset to get started.\n\n- Fix missing values\n- Detect and handle outliers\n- Detect and handle duplicates\n- Resolve feature inconsistency\n- Scale and encode data\n- Feature selection with a focus on the target variable\n- Features engineering",
  time: new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  }),
};

export const ALLOWED_MIME_TYPES = [
  "text/csv",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  "application/vnd.ms-excel",
];
