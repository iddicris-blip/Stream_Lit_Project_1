import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load a sample dataset (Iris flower dataset)
data = load_iris()
X = data.data          # Features (measurements)
y = data.target        # Labels (flower types)

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Check accuracy
predictions = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

# 5. Save the model to a file
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as model.pkl")
