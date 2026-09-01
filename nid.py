import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import matplotlib.pyplot as plt

#Load dataset
df_data = pd.read_csv(r'C:\Users\Nishant\Downloads\kdd_test.csv.zip')
df_data = df_data.rename(columns={'labels': 'label'})  # Standardizing label name

# Remove columns with only one unique value (not useful for ML)
constant_columns = [col for col in df_data.columns if df_data[col].nunique() <= 1]
df_data = df_data.drop(columns=constant_columns)
print("Removed constant columns:", constant_columns)
# Create binary target variable (0 = Normal, 1 = Attack)
df_data['target'] = (df_data['label'].str.lower() != 'normal').astype(int)
# Separate features and target
X_features = df_data.drop(columns=['label', 'target'])
y_target = df_data['target']
# Identify categorical columns
categorical_columns = X_features.select_dtypes(include=['object']).columns.tolist()
# One-Hot Encoding for categorical features
X_features = pd.get_dummies(X_features, columns=categorical_columns, drop_first=True)
# Replace infinite or missing values
X_features = X_features.replace([np.inf, -np.inf], np.nan).fillna(0)
print("Final feature set shape:", X_features.shape)
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_features,y_target,test_size=0.25,stratify=y_target,random_state=42)
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

#  Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  Train Models

logistic_model = LogisticRegression(max_iter=1000, solver='liblinear')
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

logistic_model.fit(X_train_scaled, y_train)
random_forest_model.fit(X_train_scaled, y_train)


# Predictions and Evaluation
y_pred_lr = logistic_model.predict(X_test_scaled)
y_proba_lr = logistic_model.predict_proba(X_test_scaled)[:, 1]

y_pred_rf = random_forest_model.predict(X_test_scaled)
y_proba_rf = random_forest_model.predict_proba(X_test_scaled)[:, 1]

print("\n--- Logistic Regression Report ---")
print(classification_report(y_test, y_pred_lr))

print("\n--- Random Forest Report ---")
print(classification_report(y_test, y_pred_rf))

print("ROC AUC Logistic Regression:", roc_auc_score(y_test, y_proba_lr))
print("ROC AUC Random Forest:", roc_auc_score(y_test, y_proba_rf))

#Plot ROC Curve
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {roc_auc_score(y_test, y_proba_lr):.3f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_score(y_test, y_proba_rf):.3f})")
plt.plot([0, 1], [0, 1], '--')  # Random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()

#Save Trained Models

joblib.dump(logistic_model, "logistic_model.joblib")
joblib.dump(random_forest_model, "random_forest_model.joblib")
joblib.dump(scaler, "scaler.joblib")
plt.savefig("roc_curve.png")

print("\nModels and ROC plot saved successfully.")
