# **Linear Regression (Insurance cost Prediction)**
# Firstly we Import pandas for Makes Data Handling Super Easy

import pandas as pd 

# ** Load Dataset**
# df.head is use for saw the data table

df=pd.read_csv("/content/insurance.csv")
df.head()

# **Prepare Data** (age + smoker → charges)according to the age and smoker status we predict the medical predict_charges
# Convert smoker column to numeric (yes→1, no→0)

df = df[['age', 'smoker', 'charges']]

# Convert smoker to numeric
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

#  Define features (X) and target (y)



X = df[['age', 'smoker']]
y = df['charges']

# Train/Test Split- split dataset into training part and testing part here we take test_size=0.2 .So it is devide 80% for training and 20% for testing.


# sklearn.model_selection is a module in Scikit-learn that provides tools for splitting, validating, and tuning machine learning models.
# It is a module in Scikit-learn that contains functions used for:

# Splitting data

# Cross-validation

# Hyperparameter tuning

# Model evaluation

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Imports the LinearRegression class from Scikit-learn.

# Linear Regression is a machine learning algorithm used for predicting continuous values (like price, salary, cost, etc.)

from sklearn.linear_model import LinearRegression


model = LinearRegression()
model.fit(X_train, y_train)

# These functions help evaluate how good our machine learning model is.

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# ---------------------------------------------------------
# 6. Check Model Performance
# ---------------------------------------------------------
y_pred = model.predict(X_test)
# Evaluation
print("Intercept:", model.intercept_)
print("Coefficients (age, smoker):", model.coef_)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5   # FIXED
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R² Score:", r2)


#  Helper Function → Predict Charges
# You create a function named predict_charges

# It accepts two inputs:

# age → numeric value

# smoker → "yes" or "no"

# This function will return the predicted insurance charges.


def predict_charges(age, smoker):
    smoker_value = 1 if smoker.lower() == "yes" else 0
    return model.predict([[age, smoker_value]])[0]


# here we import numpy for numerical and mathematical operations in Python.and
#  Matplotlib is the most popular plotting library in Python.

# pyplot (plt) is the module used for making graphs.

import numpy as np
import matplotlib.pyplot as plt

#  Plot Actual vs Predicted (Graph)


plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Medical Charges")
plt.show()


# Save model as Pickle



import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved successfully as model.pkl")