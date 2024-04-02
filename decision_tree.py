import time

from hyperparameters import implementation_1_hyperparameters
from preprocessing import (
    get_practical_test,
    load_data,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    f1_score,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score
)
from colorama import Fore, Style

# =============================================================================
#                           Load and preprocess data
# =============================================================================

cleaned_data = load_data()

# features, target = extract_features_target(cleaned_data)
# x_train, x_test, y_train, y_test = split_data(features, target, 0.2)

(
    x_train, x_test,
    y_train, y_test,
    x_practical_test, y_practical_test
) = get_practical_test(
    cleaned_data,
    'Winter Career Fair 2024',
    0.2
)


# =============================================================================
#                           Initialize metrics
# =============================================================================
model_count = 5
avg_accuracy, avg_mse, avg_f1, avg_recall, avg_precision = 0, 0, 0, 0, 0
(avg_practical_accuracy, avg_practical_mse,
 avg_practical_f1, avg_positive_predicted) = 0, 0, 0, 0
time_elapsed = 0
start_time = time.time()

# =============================================================================
#                           Train and evaluate models
# =============================================================================


def evaluate_model(
    model: DecisionTreeClassifier,
    x_test,
    y_test,
    start_time,
    time_elapsed
):
    time_taken = time.time() - start_time - time_elapsed
    print(f'{Fore.GREEN}  Model {i+1} trained in '
          f'{time_taken:.2f} seconds' + Style.RESET_ALL)

    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    return mse, accuracy, f1, recall, precision, time_taken


def evaluate_practical(
    model: DecisionTreeClassifier,
    x_test,
    y_test
):
    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    total_positive_predicted = sum(y_pred)

    return mse, accuracy, f1, total_positive_predicted


def print_metrics(
    mse=None,
    accuracy=None,
    f1=None,
    recall=None,
    precision=None
):
    if mse is not None:
        if mse > 0.25:
            print(f'{Fore.RED}    Mean Squared Error: '
                  f'{Fore.CYAN}{mse:.4f}' + Style.RESET_ALL)
        else:
            print(f'{Fore.GREEN}    Mean Squared Error: '
                  f'{Fore.CYAN}{mse:.4f}' + Style.RESET_ALL)

    if accuracy is not None:
        if accuracy < 0.75:
            print(f'{Fore.RED}    Accuracy: '
                  f'{Fore.CYAN}{accuracy:.4f}' + Style.RESET_ALL)
        else:
            print(f'{Fore.GREEN}    Accuracy: '
                  f'{Fore.CYAN}{accuracy:.4f}' + Style.RESET_ALL)

    if f1 is not None:
        if f1 < 0.5:
            print(f'{Fore.RED}    F1 Score: '
                  f'{Fore.CYAN}{f1:.4f}' + Style.RESET_ALL)
        else:
            print(f'{Fore.GREEN}    F1 Score: '
                  f'{Fore.CYAN}{f1:.4f}' + Style.RESET_ALL)

    if recall is not None:
        if recall < 0.6:
            print(f'{Fore.RED}    Recall: '
                  f'{Fore.CYAN}{recall:.4f}' + Style.RESET_ALL)
        else:
            print(f'{Fore.GREEN}    Recall: '
                  f'{Fore.CYAN}{recall:.4f}' + Style.RESET_ALL)
    if precision is not None:
        if precision < 0.6:
            print(f'{Fore.RED}    Precision: '
                  f'{Fore.CYAN}{precision:.4f}' + Style.RESET_ALL)
        else:
            print(f'{Fore.GREEN}    Precision: '
                  f'{Fore.CYAN}{precision:.4f}' + Style.RESET_ALL)


print(Fore.MAGENTA + f"\nTraining {model_count} models..." + Style.RESET_ALL)

for i in range(model_count):
    print(Fore.MAGENTA + f"\n  Training Model {i + 1}..." + Style.RESET_ALL)

    model = DecisionTreeClassifier(**implementation_1_hyperparameters)
    model.fit(x_train, y_train)

    (
        mse, accuracy, f1, recall, precision, time_taken
    ) = evaluate_model(
        model, x_test, y_test, start_time, time_elapsed
    )

    # print_metrics(mse, accuracy, f1, recall, precision)

    avg_mse += mse
    avg_accuracy += accuracy
    avg_f1 += f1
    avg_recall += recall
    avg_precision += precision
    time_elapsed += time_taken

    (
        mse, accuracy, f1, total_positive_predicted
    ) = evaluate_practical(
        model, x_practical_test, y_practical_test
    )

    avg_practical_mse += mse
    avg_practical_accuracy += accuracy
    avg_practical_f1 += f1
    avg_positive_predicted += total_positive_predicted


print(Fore.GREEN + f'\nAll models trained and evaluated in '
      f'{Fore.CYAN}{time_elapsed:.2f} seconds' + Style.RESET_ALL)


# =============================================================================
#                           Average metrics
# =============================================================================
print(Fore.RED + "\n--------------------------------" + Style.RESET_ALL)
print(Fore.RED + "        Average Metrics" + Style.RESET_ALL)
print(Fore.RED + "--------------------------------" + Style.RESET_ALL)

print_metrics(
    avg_mse/model_count,
    avg_accuracy/model_count,
    avg_f1/model_count,
    avg_recall/model_count,
    avg_precision/model_count
)

# =============================================================================
#                           Practical test
# =============================================================================

print(Fore.RED + "\n--------------------------------" + Style.RESET_ALL)
print(f'{Fore.MAGENTA}     Practical Test Results' + Style.RESET_ALL)
print(Fore.RED + "--------------------------------" + Style.RESET_ALL)

total_positive_actual = sum(y_practical_test)

print(f'{Fore.LIGHTYELLOW_EX}  Average Predicted Attendance: '
      f'{Fore.CYAN}{avg_positive_predicted/model_count:.1f}' + Style.RESET_ALL)
print(f'{Fore.LIGHTYELLOW_EX}  Actual Attendance: '
      f'{Fore.CYAN}{total_positive_actual}' + Style.RESET_ALL)
print(f'{Fore.LIGHTYELLOW_EX}  Total Students: '
      f'{Fore.CYAN}{len(y_practical_test)}{Style.RESET_ALL}\n')

print_metrics(
    avg_practical_mse/model_count,
    avg_practical_accuracy/model_count,
    avg_practical_f1/model_count
)

print()
