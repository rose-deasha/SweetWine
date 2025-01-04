from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

# Load datasets for red and white wines
print("Loading datasets...")
red_wine = pd.read_csv(
    '/Users/jamikapage/Library/Mobile Documents/com~apple~CloudDocs/Portfolio/SweetWines/winequality-red.csv',
    delimiter=';'
)
white_wine = pd.read_csv(
    '/Users/jamikapage/Library/Mobile Documents/com~apple~CloudDocs/Portfolio/SweetWines/winequality-white.csv',
    delimiter=';'
)
print("Datasets loaded successfully!")

# Add 'wine_type' column
print("Adding wine type labels...")
red_wine['wine_type'] = 1  # Red wine
white_wine['wine_type'] = 0  # White wine

# Combine datasets
print("Combining datasets...")
data = pd.concat([red_wine, white_wine], axis=0)
print(f"Combined dataset shape: {data.shape}")

# Define sweetness threshold
threshold = 20
data['is_sweet'] = (data['residual sugar'] > threshold).astype(int)
print(f"'is_sweet' value counts:\n{data['is_sweet'].value_counts()}")

# Select features and target variable
print("Preparing feature set and target variable...")
X = data[
    [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "wine_type",
    ]
]
y = data['is_sweet']
print(f"Feature set shape: {X.shape}, Target variable shape: {y.shape}")

# Train/test split
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# Apply SMOTE
print("Applying SMOTE to balance the dataset...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print("SMOTE applied. New target variable distribution:")
print(pd.Series(y_resampled).value_counts())

# Train a Random Forest model
print("Training Random Forest model...")
model = RandomForestClassifier(random_state=42)
model.fit(X_resampled, y_resampled)
print("Model training complete!")

# Evaluate the model
print("Evaluating the model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Sweet', 'Sweet'], yticklabels=['Not Sweet', 'Sweet'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heatmap')
plt.show()

# Get feature importances from the trained model
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='bar', color='skyblue')
plt.title('Feature Importances')
plt.ylabel('Importance Score')
plt.xlabel('Features')
plt.xticks(rotation=45)
plt.show()

# Class distribution before SMOTE
plt.figure(figsize=(8, 4))
data['is_sweet'].value_counts().plot(kind='bar', color=['skyblue', 'orange'])
plt.title('Class Distribution Before SMOTE')
plt.xlabel('Classes (0 = Not Sweet, 1 = Sweet)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Class distribution after SMOTE
plt.figure(figsize=(8, 4))
pd.Series(y_resampled).value_counts().plot(kind='bar', color=['skyblue', 'orange'])
plt.title('Class Distribution After SMOTE')
plt.xlabel('Classes (0 = Not Sweet, 1 = Sweet)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

from sklearn.metrics import precision_recall_curve

# Get precision and recall values
precision, recall, thresholds = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])

# Plot the Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', color='darkorange')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='residual sugar', hue='is_sweet', kde=True, palette=['blue', 'orange'], bins=30)
plt.title('Residual Sugar Distribution by Sweetness')
plt.xlabel('Residual Sugar')
plt.ylabel('Count')
plt.legend(['Not Sweet', 'Sweet'])
plt.show()


