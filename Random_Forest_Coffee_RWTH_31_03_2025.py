import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load your dataset
data = pd.read_csv("coffee_suppliers_with_coords.csv")

# Define target before encoding
y = data['Sustainability_Risk']

# Drop non-feature columns before encoding
X_raw = data.drop(['Supplier', 'Sustainability_Risk'], axis=1)

# Encode categorical variables
X = pd.get_dummies(X_raw, drop_first=True)
from sklearn.utils import resample

# Combine features and target
df_full = pd.concat([X, y], axis=1)

# Separate by class
df_high = df_full[df_full['Sustainability_Risk'] == 'High']
df_medium = df_full[df_full['Sustainability_Risk'] == 'Medium']
df_low = df_full[df_full['Sustainability_Risk'] == 'Low']

# Upsample the 'High' class
df_high_upsampled = resample(
    df_high,
    replace=True,
    n_samples=20,  # Adjust this number as needed
    random_state=42
)

# Combine all
df_balanced = pd.concat([df_medium, df_low, df_high_upsampled])

# Shuffle (optional but good practice)
df_balanced = df_balanced.sample(frac=1, random_state=42)

# Separate again
X = df_balanced.drop('Sustainability_Risk', axis=1)
y = df_balanced['Sustainability_Risk']

from sklearn.model_selection import cross_val_score

# Define model
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

# Perform 5-fold cross-validation
scores = cross_val_score(rf_model, X, y, cv=5)

# Print accuracy results
print(f"Cross-validated accuracy: {scores.mean():.2f} ± {scores.std():.2f}")



# Re-fit the model on the full balanced dataset
rf_model.fit(X, y)

# Calculate feature importances
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

# Print top features
print("\nTop 10 Feature Importances:")
print(importances.head(10))

# Optional: Plot feature importances
import matplotlib.pyplot as plt

importances.head(10).plot(kind='barh', title='Top 10 Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


