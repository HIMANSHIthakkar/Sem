# Sem
# -----------------------------
# STEP 1: Upload CSV File
# -----------------------------
from google.colab import files
import pandas as pd

uploaded = files.upload()   # Opens file picker

filename = list(uploaded.keys())[0]   # Get uploaded file name
df = pd.read_csv(filename)            # Load CSV file

print("Dataset Loaded Successfully!")
print(df.head())                      # Show first 5 rows


# -----------------------------
# STEP 2: Prepare X and y
# -----------------------------
# IMPORTANT:
# Replace 'target' with your actual target column name
# Example: 'price', 'marks', 'output'

X = df.drop("Outcome", axis=1).values   # Features
y = df["Outcome"].values                # Target

print("Shapes:", X.shape, y.shape)


# -----------------------------
# STEP 3: Apply Linear Regression
# -----------------------------
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)

# Predict using the first row
prediction = model.predict([X[0]])

print("\nPrediction for first row:", prediction)
print("Actual value:", y[0])
-----------------------------------------------------------------------------------------------------------------------------------------


from google.colab import files
import pandas as pd

# Upload diabetes.csv
uploaded = files.upload()

df = pd.read_csv("diabetes.csv")
print(df.head())

# Features and target
X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

print("Prediction for first row:", model.predict([X_test[0]]))
print("Actual output:", y_test[0])

print("Accuracy:", model.score(X_test, y_test))
------------------------------------------------------------------------------------------------------------------------------------------------

SVM


from sklearn.svm import SVC

svm = SVC()
svm.fit(X_train, y_train)

print("Prediction:", svm.predict([X_test[0]]))
print("Accuracy:", svm.score(X_test, y_test))
-----------------------------------------------------------------------------------------------------------------------------------------------------


mlp


from tensorflow import keras
from keras import layers

model = keras.Sequential([
    layers.Dense(12, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')  
    # binary output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, verbose=0)

print(model.predict(X_test[:5]))
--------------------------------------------------------------------------------------------------------------------------------------------------------------

