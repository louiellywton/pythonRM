from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import pandas as pd

# Dataset sample
# data = {
#     'person_id': [1, 2, 3],
#     'gender': ['male', 'female', 'male'],
#     'age': [27, 22, 35],
#     'occupation': ['Software engineer', 'Student', 'Data scientist'],
#     'sleep_duration': [6.1, 7.5, 5.8],
#     'sleep_quality': [6, 8, 5],
#     'physical_activity': [42, 30, 45],
#     'stress_level': [6, 4, 7],
#     'bmi_category': ['overweight', 'normal', 'overweight'],
#     'blood_pressure': ['126/83', '118/72', '130/90']
# }

df = pd.read_csv('venv/Sleep_health_and_lifestyle_dataset.csv', nrows=50)

# Feature selection
features = ['Age', 'Physical Activity Level', 'Stress Level']
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

# Categorizing sleep quality
def categorize_sleep_quality(value):
    if value < 6:
        return 'Bad'
    elif value < 8:
        return 'Normal'
    else:
        return 'Good'

# Applying categorization to actual and predicted values
y_test_category = y_test.apply(categorize_sleep_quality)
predictions_category = pd.Series(predictions).apply(categorize_sleep_quality)

# Evaluating the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Visualizing the results
plt.figure(figsize=(10, 6))

plt.scatter(X_test['Age'], y_test, color='black', label='Actual')
plt.scatter(X_test['Age'], predictions, color='blue', label='Predicted')
plt.xlabel('Age')
plt.ylabel('Sleep Quality')
plt.title('Actual vs. Predicted Sleep Quality')
plt.legend()

# Adding categorization information to the plot
for i, txt in enumerate(y_test_category):
    plt.annotate(txt, (X_test['Age'].iloc[i], y_test.iloc[i]), textcoords="offset points", xytext=(0,10), ha='center')

for i, txt in enumerate(predictions_category):
    plt.annotate(txt, (X_test['Age'].iloc[i], predictions[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.show()
