import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

# Dataset Sample
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

with open('venv/Sleep_health_and_lifestyle_dataset.csv','r') as csv_file:
    csv_reader = csv.reader(csv_file)
    
    for line in csv_reader:
        print(line)
        
df = pd.read_csv('venv/Sleep_health_and_lifestyle_dataset.csv', nrows=50)

# Function to classify age and calculate Mental Quality
def calculate_mental_quality(row):
    # Classify age
    age = row['Age']
    if age <= 19:
        age_points = 6
    elif 20 <= age <= 59:
        age_points = 5
    else:
        age_points = 3

    # Classify physical activity
    physical_activity = row['Physical Activity Level']
    if physical_activity < 40:
        activity_points = 6
    elif 40 <= physical_activity <= 45:
        activity_points = 9
    elif 46 <= physical_activity <= 50:
        activity_points = 10
    elif 51 <= physical_activity <= 55:
        activity_points = 8
    elif 56 <= physical_activity <= 60:
        activity_points = 7
    elif physical_activity > 80:
        activity_points = 3
    else:
        activity_points = 5

    # Stress level points
    stress_points = row['Stress Level']

    # Blood pressure classification
    blood_pressure = row['Blood Pressure']
    systolic_bp = int(blood_pressure.split('/')[0])
    if systolic_bp < 120:
        bp_points = 9
    elif 120 <= systolic_bp <= 129:
        bp_points = 7
    else:
        bp_points = 5

    # BMI category points
    bmi_category = row['BMI Category']
    if bmi_category == 'normal':
        bmi_points = 10
    elif bmi_category == 'normal weight':
        bmi_points = 8
    elif bmi_category == 'overweight':
        bmi_points = 6
    else:
        bmi_points = 4

    # Calculate Mental Quality
    mental_quality = (age_points + activity_points + stress_points + bp_points + bmi_points) / 5

    # Scaling the result to a 1-10 range
    scaled_mental_quality = np.clip(mental_quality, 1, 10)

    return scaled_mental_quality

# Function to classify sleep quality
def classify_sleep_quality(value):
    if value == 10:
        return 'Perfect'
    elif value >= 8:
        return 'Good'
    elif value >= 6:
        return 'Normal'
    elif value >= 4:
        return 'Bad'
    else:
        return 'Worst'

# Function to calculate sleep quality
def calculate_sleep_quality(row):
    # Sleep duration classification
    sleep_duration = int(row['Sleep Duration'])
    if sleep_duration >= 8:
        duration_points = 10
    elif sleep_duration >= 7:
        duration_points = 8
    elif sleep_duration >= 6:
        duration_points = 6
    elif sleep_duration >= 5:
        duration_points = 4
    else:
        duration_points = 2

    # Stress level points
    stress_points = row['Stress Level']

    # Physical activity classification
    physical_activity = row['Physical Activity Level']
    if physical_activity < 40:
        activity_points = 6
    elif 40 <= physical_activity <= 45:
        activity_points = 9
    elif 46 <= physical_activity <= 50:
        activity_points = 10
    elif 51 <= physical_activity <= 55:
        activity_points = 8
    elif 56 <= physical_activity <= 60:
        activity_points = 7
    elif physical_activity > 80:
        activity_points = 3
    else:
        activity_points = 5

    # Calculate Sleep Quality
    sleep_quality = (duration_points + stress_points + activity_points) / 3

    # Scaling the result to a 1-10 range
    scaled_sleep_quality = np.clip(sleep_quality, 1, 10)

    # Classify sleep quality
    classified_sleep_quality = classify_sleep_quality(scaled_sleep_quality)

    return scaled_sleep_quality, classified_sleep_quality

# Applying the function to the dataset
df['mental_quality'] = df.apply(calculate_mental_quality, axis=1)

# Applying the function to the dataset
df['sleep_quality'], df['sleep_quality_category'] = zip(*df.apply(calculate_sleep_quality, axis=1))

# Displaying the dataset with calculated Mental Quality
print(df[['Person ID', 'sleep_quality', 'mental_quality']])

# Bar graph for Sleep Quality and Mental Quality
fig, ax = plt.subplots(figsize=(8, 6))

# Bar width
bar_width = 0.35

# Bar positions
bar_positions1 = np.arange(len(df['Person ID']))
bar_positions2 = bar_positions1 + bar_width

# Bar plots
bars1 = ax.bar(bar_positions1, df['sleep_quality'], width=bar_width, label='Sleep Quality')
bars2 = ax.bar(bar_positions2, df['mental_quality'], width=bar_width, label='Mental Quality')

# Adding labels and title
ax.set_xlabel('Person ID')
ax.set_ylabel('Score')
ax.set_title('Sleep Quality and Mental Quality Comparison')
ax.set_xticks(bar_positions1 + bar_width / 2)
ax.set_xticklabels(df['Person ID'])
ax.legend()

# Displaying the bar graph
plt.show()
