import pandas as pd
import random

# Load existing data (if you have any)
# df = pd.read_csv('intents_expanded.csv')
# Or start fresh
df = pd.DataFrame(columns=['prompt', 'intent'])

# Define augmentation templates and variations for each intent
augmentation_rules = {
    'handle_missing_values': {
        'verbs': ['handle', 'fill', 'impute', 'replace', 'deal with', 'fix', 'address', 'manage', 'process', 'treat'],
        'objects': ['missing values', 'NaNs', 'null values', 'empty cells', 'NA values', 'missing data', 
                   'nulls', 'blank entries', 'empty entries', 'missing entries', 'undefined values', 'missing cells'],
        'methods': ['using mean', 'with median', 'using mode', 'by forward fill', 'with interpolation',
                   'using mean value', 'with median value', 'by mode', 'via mean imputation', 'with ffill',
                   'by backward fill', 'using linear interpolation', 'with constant value', 'by dropping rows'],
        'locations': ['in the dataset', 'in dataframe', 'in column {col}', 'for column {col}', 
                     'in the data', 'across all columns', 'in numeric columns', 'from {col}', 'in {col} column'],
        'templates': [
            '{verb} {object}',
            '{verb} {object} {method}',
            '{verb} {object} {location}',
            '{verb} {object} {method} {location}',
            '{object} should be {verb}',
            '{verb} all {object}',
            '{method} for {object}',
            '{verb} {object} in data',
        ]
    },
    
    'detect_outliers': {
        'verbs': ['detect', 'find', 'identify', 'remove', 'delete', 'drop', 'eliminate', 'filter out', 
                 'exclude', 'discard', 'get rid of', 'clean'],
        'objects': ['outliers', 'anomalies', 'extreme values', 'outlier records', 'anomalous data', 
                   'extreme observations', 'outlier entries', 'unusual values', 'extreme points', 'abnormal data',
                   'outlier rows', 'extreme data points'],
        'methods': ['using IQR', 'with z-score', 'by statistical method', 'using standard deviation',
                   'with isolation forest', 'using boxplot method', 'by percentile', 'using 3-sigma rule',
                   'with MAD method', 'using DBSCAN'],
        'locations': ['in the dataset', 'from dataframe', 'in column {col}', 'from the data',
                     'across all features', 'in numeric columns', 'from {col}', 'in {col} column'],
        'templates': [
            '{verb} {object}',
            '{verb} {object} {method}',
            '{verb} {object} {location}',
            '{verb} {object} {method} {location}',
            '{verb} and {verb2} {object}',
            '{method} to {verb} {object}',
            '{verb} all {object}',
        ]
    },
    
    'keep_outliers': {
        'verbs': ['keep', 'preserve', 'retain', 'maintain', "don't remove", "don't delete", 'leave', 'save'],
        'objects': ['outliers', 'extreme values', 'anomalies', 'outlier records', 'anomalous data',
                   'unusual values', 'extreme points', 'abnormal data', 'outlier rows'],
        'reasons': ['for analysis', 'for study', 'untouched', 'in dataset', 'as they are', 'unchanged',
                   'for review', 'for investigation', 'intact', 'for further analysis'],
        'templates': [
            '{verb} {object}',
            '{verb} {object} {reason}',
            "do not remove {object}",
            '{object} should be {verb}',
            '{verb} all {object}',
            "don't {verb2} {object}",
            '{object} must be {verb}',
        ]
    },
    
    'remove_duplicates': {
        'verbs': ['remove', 'drop', 'delete', 'eliminate', 'filter', 'deduplicate', 'clean', 'get rid of', 'discard'],
        'objects': ['duplicates', 'duplicate rows', 'repeated entries', 'duplicate records', 
                   'identical rows', 'repeated records', 'duplicate entries', 'redundant rows', 'duplicate data',
                   'repeated rows', 'duplicate observations'],
        'locations': ['from dataset', 'in dataframe', 'by column {col}', 'based on {col}',
                     'from the data', 'in the table', 'using {col}', 'considering {col}', 'from data'],
        'templates': [
            '{verb} {object}',
            '{verb} {object} {location}',
            '{verb} all {object}',
            '{object} should be {verb}',
            '{verb} {object} from data',
            '{location} {verb} {object}',
        ]
    },
    
    'encode_categorical': {
        'verbs': ['encode', 'transform', 'convert', 'map', 'change', 'turn', 'translate'],
        'objects': ['categorical variables', 'categories', 'categorical columns', 'text columns',
                   'string categories', 'categorical features', 'category columns', 'categorical data',
                   'non-numeric columns', 'text features', 'string variables'],
        'methods': ['using label encoding', 'with one-hot encoding', 'to numeric', 'to integers',
                   'to dummy variables', 'using ordinal encoding', 'with binary encoding', 'to numerical values',
                   'using target encoding', 'with frequency encoding'],
        'locations': ['in column {col}', 'for {col}', 'in the dataset', 'in dataframe', 'in {col} column'],
        'templates': [
            '{verb} {object}',
            '{verb} {object} {method}',
            'apply {method} to {object}',
            '{verb} all {objects}',
            '{method} for {object}',
            '{verb} {object} {location}',
            '{verb} {object} {method} {location}',
        ]
    },
    
    'feature_selection': {
        'verbs': ['select', 'choose', 'pick', 'identify', 'find', 'extract', 'determine', 'get'],
        'objects': ['features', 'important features', 'best features', 'top features', 'attributes',
                   'predictors', 'relevant features', 'key features', 'significant features', 'useful features',
                   'top predictors', 'best attributes'],
        'methods': ['using importance', 'by correlation', 'top-k method', 'by variance',
                   'using mutual information', 'with statistical tests', 'using chi-square', 'by RFE',
                   'with filter method', 'using wrapper method', 'by embedded method'],
        'purposes': ['for modeling', 'for training', 'for prediction', 'for the model', 'for ML',
                    'for classification', 'for regression', 'for analysis'],
        'numbers': ['top 10', 'top 5', 'best 20', 'top-k', 'most important'],
        'templates': [
            '{verb} {object}',
            '{verb} {object} {method}',
            '{verb} {object} {purpose}',
            'perform feature selection',
            'run feature selection {method}',
            '{verb} {number} {object}',
            '{method} to {verb} {object}',
            '{verb} {object} {method} {purpose}',
        ]
    }
}

# Column name variations for templates
column_names = ['age', 'price', 'salary', 'id', 'user_id', 'customer_id', 'amount', 'score', 
                'name', 'category', 'date', 'income', 'years_experience', 'rating']

def generate_phrase(intent, rules):
    """Generate a single phrase for an intent"""
    template = random.choice(rules['templates'])
    
    # Build the phrase
    phrase = template
    
    if '{verb}' in phrase:
        phrase = phrase.replace('{verb}', random.choice(rules['verbs']))
    if '{verb2}' in phrase:
        phrase = phrase.replace('{verb2}', random.choice(rules['verbs']))
    if '{object}' in phrase:
        phrase = phrase.replace('{object}', random.choice(rules['objects']))
    if '{objects}' in phrase:
        phrase = phrase.replace('{objects}', random.choice(rules['objects']))
    if '{method}' in phrase and 'methods' in rules:
        phrase = phrase.replace('{method}', random.choice(rules['methods']))
    if '{location}' in phrase and 'locations' in rules:
        location = random.choice(rules['locations'])
        if '{col}' in location:
            location = location.replace('{col}', random.choice(column_names))
        phrase = phrase.replace('{location}', location)
    if '{reason}' in phrase and 'reasons' in rules:
        phrase = phrase.replace('{reason}', random.choice(rules['reasons']))
    if '{purpose}' in phrase and 'purposes' in rules:
        phrase = phrase.replace('{purpose}', random.choice(rules['purposes']))
    if '{number}' in phrase and 'numbers' in rules:
        phrase = phrase.replace('{number}', random.choice(rules['numbers']))
    
    return phrase.strip()

# Generate augmented data
augmented_data = []
target_per_intent = 500

# Only use the 6 specified intents
target_intents = [
    'handle_missing_values',
    'detect_outliers', 
    'keep_outliers',
    'remove_duplicates',
    'encode_categorical',
    'feature_selection'
]

for intent in target_intents:
    print(f"Generating data for {intent}...")
    
    # Get existing prompts for this intent (if any)
    if not df.empty and intent in df['intent'].values:
        existing_prompts = set(df[df['intent'] == intent]['prompt'].tolist())
        augmented_data.extend([(p, intent) for p in existing_prompts])
    else:
        existing_prompts = set()
    
    if intent in augmentation_rules:
        rules = augmentation_rules[intent]
        attempts = 0
        max_attempts = target_per_intent * 20  # Prevent infinite loop
        
        while len([d for d in augmented_data if d[1] == intent]) < target_per_intent and attempts < max_attempts:
            phrase = generate_phrase(intent, rules)
            
            # Check if phrase is unique
            if phrase not in existing_prompts and phrase not in [d[0] for d in augmented_data]:
                augmented_data.append((phrase, intent))
            
            attempts += 1
    
    current_count = len([d for d in augmented_data if d[1] == intent])
    print(f"  ✅ Generated {current_count} phrases for {intent}")

# Create final dataframe
augmented_df = pd.DataFrame(augmented_data, columns=['prompt', 'intent'])

# Shuffle the data
augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to new file
augmented_df.to_csv('intents_augmented.csv', index=False)

print(f"\n🎉 Augmentation complete!")
print(f"Total samples: {len(augmented_df)}")
print("\n📊 Samples per intent:")
print(augmented_df['intent'].value_counts().sort_index())
print("\n💾 Saved to: intents_augmented.csv")

# Show some examples
print("\n🔍 Sample generated prompts:")
for intent in target_intents:
    print(f"\n{intent}:")
    samples = augmented_df[augmented_df['intent'] == intent].head(5)
    for prompt in samples['prompt']:
        print(f"  - {prompt}")