import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('venv/Sleep_health_and_lifestyle_dataset.csv', nrows=50)

# Feature selection
features = ['Age', 'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps']
X = df[features]
y = df['Quality of Sleep']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a Linear Regression model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Making predictions on the test set
predictions = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

import matplotlib.pyplot as plt

plt.scatter(X_test['Age'], y_test, color='black', label='Actual')
plt.scatter(X_test['Age'], predictions, color='blue', label='Predicted')
plt.xlabel('Age')
plt.ylabel('Quality of Sleep')
plt.legend()
plt.show()
