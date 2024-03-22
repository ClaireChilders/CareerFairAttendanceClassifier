import time
from hyperparameters import implementation_1_hyperparameters
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    f1_score,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score
)
from colorama import Fore, Style

# Load and Merge Data

print(Fore.MAGENTA + "\nLoading data..." + Style.RESET_ALL)

appointment_df = pd.read_csv('data/appointment_data.csv')
attendance_df = pd.read_csv('data/attendance_data.csv')
career_fair_df = pd.read_csv('data/career_fair_data.csv')
registration_df = pd.read_csv('data/registration_data.csv')
student_df = pd.read_csv('data/student_data.csv')

print(Fore.GREEN + "Data loaded" + Style.RESET_ALL)

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

print(Fore.GREEN + "Data merged" + Style.RESET_ALL)
print(Fore.MAGENTA + "\nCleaning data..." + Style.RESET_ALL)

# Convert Yes/No to 1/0

merged_data['is_checked_in'] = merged_data['is_checked_in'].apply(
    lambda x: 1 if x == 'Yes' else 0)
merged_data['is_pre_registered'] = merged_data['is_pre_registered'].apply(
    lambda x: 1 if x == 'Yes' else 0)
merged_data['stu_is_activated'] = merged_data['stu_is_activated'].apply(
    lambda x: 1 if x == 'Yes' else 0)
merged_data['stu_is_visible'] = merged_data['stu_is_visible'].apply(
    lambda x: 1 if x == 'Yes' else 0)

print(Fore.BLUE + "  Yes/No converted to 1/0" + Style.RESET_ALL)

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

print(Fore.BLUE + "  Strings converted to integers" + Style.RESET_ALL)

# Convert dates to year only
# stu_created, stu_login, stu_grad_date

merged_data['stu_created'] = pd.to_datetime(
    merged_data['stu_created']).dt.year
merged_data['stu_login'] = pd.to_datetime(
    merged_data['stu_login']).dt.year
merged_data['stu_grad_date'] = pd.to_datetime(
    merged_data['stu_grad_date']).dt.year

print(Fore.BLUE + "  Dates converted to year only" + Style.RESET_ALL)

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

print(
    Fore.BLUE + "  Categorical features converted to dummies" + Style.RESET_ALL
)

target = merged_data['is_checked_in']

print(Fore.GREEN + "Data cleaned" + Style.RESET_ALL)

print(Fore.MAGENTA + "\nSplitting data..." + Style.RESET_ALL)
print(Fore.BLUE + "  Test size: 0.2" + Style.RESET_ALL)

x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=0)

print(Fore.GREEN + "Data split" + Style.RESET_ALL)

model_count = 5
avg_accuracy = 0
avg_mse = 0
avg_f1 = 0
avg_recall = 0
avg_precision = 0

time_elapsed = 0
start_time = time.time()

print(Fore.MAGENTA + f"\nTraining {model_count} models..." + Style.RESET_ALL)

for i in range(model_count):
    print(Fore.MAGENTA + f"\n  Training Model {i + 1}..." + Style.RESET_ALL)

    model = DecisionTreeClassifier(**implementation_1_hyperparameters)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    time_taken = time.time() - start_time - time_elapsed
    time_elapsed += time_taken

    print(f'{Fore.GREEN}  Model {i+1} trained in '
          f'{time_taken:.2f} seconds' + Style.RESET_ALL)

    mse = mean_squared_error(y_test, y_pred)
    print(f'{Fore.BLUE}    Mean Squared Error: '
          f'{Fore.CYAN}{mse:.4f}' + Style.RESET_ALL)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'{Fore.BLUE}    Accuracy: '
          f'{Fore.CYAN}{accuracy:.4f}' + Style.RESET_ALL)

    f1 = f1_score(y_test, y_pred)
    print(f'{Fore.BLUE}    F1 Score: '
          f'{Fore.CYAN}{f1:.4f}' + Style.RESET_ALL)

    recall = recall_score(y_test, y_pred)
    print(f'{Fore.BLUE}    Recall: '
          f'{Fore.CYAN}{recall:.4f}' + Style.RESET_ALL)

    precision = precision_score(y_test, y_pred)
    print(f'{Fore.BLUE}    Precision: '
          f'{Fore.CYAN}{precision:.4f}' + Style.RESET_ALL)

    avg_accuracy += accuracy
    avg_mse += mse
    avg_f1 += f1
    avg_recall += recall
    avg_precision += precision

print(Fore.GREEN + f'\nAll models trained and evaluated in '
      f'{Fore.CYAN}{time_elapsed:.2f} seconds' + Style.RESET_ALL)

print(Fore.RED + "\n--------------------------------" + Style.RESET_ALL)
print(Fore.RED + "        Average Metrics" + Style.RESET_ALL)
print(Fore.RED + "--------------------------------" + Style.RESET_ALL)

print(f'{Fore.BLUE}  Mean Squared Error: '
      f'{Fore.CYAN}{avg_mse/model_count:.4f}' + Style.RESET_ALL)
print(f'{Fore.BLUE}  Accuracy: '
      f'{Fore.CYAN}{avg_accuracy/model_count:.4f}' + Style.RESET_ALL)
print(f'{Fore.BLUE}  F1 Score: '
      f'{Fore.CYAN}{avg_f1/model_count:.4f}' + Style.RESET_ALL)
print(f'{Fore.BLUE}  Recall: '
      f'{Fore.CYAN}{avg_recall/model_count:.4f}' + Style.RESET_ALL)
print(f'{Fore.BLUE}  Precision: '
      f'{Fore.CYAN}{avg_precision/model_count:.4f}' + Style.RESET_ALL)
