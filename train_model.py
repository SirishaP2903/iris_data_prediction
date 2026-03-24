import numpy as np
import pandas as pd
from pandas.io import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import pickle

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)
print(X_test)
print(y_pred)

with open('iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('iris_scaler.pkl', 'wb') as f:
    pickle.dump(scaler,f)

print("Successfully saved model and  the  scaler")


