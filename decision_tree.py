import time
from hyperparameters import implementation_1_hyperparameters
from preprocessing import (
    load_data,
    clean_data,
    extract_features_target,
    split_data
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
data = load_data()
cleaned_data = clean_data(data)
features, target = extract_features_target(data)
x_train, x_test, y_train, y_test = split_data(features, target, 0.2)

# =============================================================================
#                           Initialize metrics
# =============================================================================
model_count = 5
avg_accuracy, avg_mse, avg_f1, avg_recall, avg_precision = 0, 0, 0, 0, 0
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


def print_metrics(mse, accuracy, f1, recall, precision):
    print(f'{Fore.BLUE}    Mean Squared Error: '
          f'{Fore.CYAN}{mse:.4f}' + Style.RESET_ALL)
    print(f'{Fore.BLUE}    Accuracy: '
          f'{Fore.CYAN}{accuracy:.4f}' + Style.RESET_ALL)
    print(f'{Fore.BLUE}    F1 Score: '
          f'{Fore.CYAN}{f1:.4f}' + Style.RESET_ALL)
    print(f'{Fore.BLUE}    Recall: '
          f'{Fore.CYAN}{recall:.4f}' + Style.RESET_ALL)
    print(f'{Fore.BLUE}    Precision: '
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

    print_metrics(mse, accuracy, f1, recall, precision)

    avg_mse += mse
    avg_accuracy += accuracy
    avg_f1 += f1
    avg_recall += recall
    avg_precision += precision
    time_elapsed += time_taken

print(Fore.GREEN + f'\nAll models trained and evaluated in '
      f'{Fore.CYAN}{time_elapsed:.2f} seconds' + Style.RESET_ALL)


# =============================================================================
#                           Average metrics
# =============================================================================
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
