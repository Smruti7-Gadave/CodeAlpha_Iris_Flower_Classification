import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("iris.csv")

print("First 5 rows:\n", df.head())
print("\nColumns:\n", df.columns)

# =========================
# REMOVE ID COLUMN (IMPORTANT FIX)
# =========================
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# =========================
# SPLIT DATA
# =========================
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

labels = y.unique()

# Convert labels to numeric
y = y.astype('category').cat.codes

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# TRAIN MODEL
# =========================
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# =========================
# PREDICT
# =========================
y_pred = model.predict(X_test)

# =========================
# EVALUATION
# =========================
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy: {:.2f}%".format(accuracy * 100))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=labels))

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================
# FEATURE IMPORTANCE
# =========================
importance = model.feature_importances_

plt.barh(X.columns, importance)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.show()

# =========================
# MANUAL INPUT TEST (FIXED)
# =========================
print("\n--- Test with Manual Input ---")

sepal_length = float(input("Enter Sepal Length: "))
sepal_width = float(input("Enter Sepal Width: "))
petal_length = float(input("Enter Petal Length: "))
petal_width = float(input("Enter Petal Width: "))

input_data = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=X.columns
)

prediction = model.predict(input_data)

print("\n Predicted Flower:", labels[prediction[0]])
