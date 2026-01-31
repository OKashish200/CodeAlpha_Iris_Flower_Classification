import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the data using the URL
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# 2. Prepare Data (Features and Target)
# The dataset has columns: sepal_length, sepal_width, petal_length, petal_width, species
X = df.drop('species', axis=1) 
y = df['species']

# 3. Split into Training and Testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Model (Classification)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 5. Evaluate Performance
predictions = model.predict(X_test)
print(f"Accuracy Score: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, predictions))
