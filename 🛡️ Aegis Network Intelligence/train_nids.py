import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Define Dataset Columns (NSL-KDD Standard Format)
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
]

# 2. Automatically Find Dataset File
possible_files = ['KDDTrain+.txt', 'KDDTrain+.csv', 'KDDTrain.txt', 'KDDTrain.csv']
dataset_path = None

for file in possible_files:
    if os.path.exists(file):
        dataset_path = file
        break

if not dataset_path:
    print("❌ Error: Dataset file nahi mili!")
    print("Kripya 'KDDTrain+.txt' file ko apne project folder me rakhein.")
    exit()

print(f"1. Loading Dataset from '{dataset_path}'...")
df = pd.read_csv(dataset_path, header=None, names=columns)

# Drop unused metadata column if present
if 'difficulty_level' in df.columns:
    df.drop('difficulty_level', axis=1, inplace=True)

# Binary Classification Mapping: 'normal' = 0, 'attack' = 1
df['target'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
df.drop('label', axis=1, inplace=True)

print("2. Preprocessing & Encoding Data...")
encoders = {}
categorical_cols = ['protocol_type', 'service', 'flag']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Split features (X) and target label (y)
X = df.drop('target', axis=1)
y = df['target']

# Train / Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("3. Training Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

print("4. Evaluating Model Performance...")
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("5. Saving Model Artifacts...")
joblib.dump(model, 'nids_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoders, 'encoders.pkl')

print("✅ Model, scaler, aur encoders kamyabi se save ho gaye hain!")
