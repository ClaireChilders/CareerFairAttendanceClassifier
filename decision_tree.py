import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error

# Load and Merge Data

print("Loading data...")

appointment_df = pd.read_csv('data/appointment_data.csv')
attendance_df = pd.read_csv('data/attendance_data.csv')
career_fair_df = pd.read_csv('data/career_fair_data.csv')
registration_df = pd.read_csv('data/registration_data.csv')
student_df = pd.read_csv('data/student_data.csv')

print("Merging data...")

merged_data = pd.merge(student_df, registration_df,
                       on='stu_id', how='left')
merged_data = pd.merge(merged_data, appointment_df,
                       on='stu_id', how='left')
merged_data = pd.merge(
    merged_data,
    attendance_df,
    on=[
        'stu_id', 'career_fair_name', 'career_fair_date'
    ],
    how='left'
)

print("Cleaning data...")

# Convert Yes/No to 1/0

merged_data['is_checked_in'] = merged_data['is_checked_in'].apply(
    lambda x: 1 if x == 'Yes' else 0)
merged_data['is_pre_registered'] = merged_data['is_pre_registered'].apply(
    lambda x: 1 if x == 'Yes' else 0)
merged_data['stu_is_activated'] = merged_data['stu_is_activated'].apply(
    lambda x: 1 if x == 'Yes' else 0)
merged_data['stu_is_visible'] = merged_data['stu_is_visible'].apply(
    lambda x: 1 if x == 'Yes' else 0)

# Convert any strings in the form '1,000' to integers
# while keeping existing integers
merged_data['appointment_count'] = merged_data['appointment_count'].apply(
    lambda x: int(x.replace(',', '')) if isinstance(x, str) else x)
merged_data['stu_applications'] = merged_data['stu_applications'].apply(
    lambda x: int(x.replace(',', '')) if isinstance(x, str) else x)
merged_data['stu_logins'] = merged_data['stu_logins'].apply(
    lambda x: int(x.replace(',', '')) if isinstance(x, str) else x)
merged_data['stu_attendances'] = merged_data['stu_attendances'].apply(
    lambda x: int(x.replace(',', '')) if isinstance(x, str) else x)

# Convert dates to year only
# stu_created, stu_login, stu_grad_date

merged_data['stu_created'] = pd.to_datetime(
    merged_data['stu_created']).dt.year
merged_data['stu_login'] = pd.to_datetime(
    merged_data['stu_login']).dt.year
merged_data['stu_grad_date'] = pd.to_datetime(
    merged_data['stu_grad_date']).dt.year

features = pd.get_dummies(
    merged_data.drop(
        ['stu_id', 'is_checked_in'], axis=1
    ), columns=[
        'appointment_types',
        'career_fair_name',
        'career_fair_date',
        'stu_school_year',
        'stu_majors',
        'stu_colleges'
    ]
)
target = merged_data['is_checked_in']

print("Splitting data into training and test sets...")

x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=0)

print("Training model...")

model = DecisionTreeClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("Model trained!")

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Square Error: {mse}')
