from hyperparameters import implementation_1_hyperparameters
from time import time
from preprocessing import (
    get_practical_test,
    load_data,
    clean_data,
    # extract_features_target,
    # split_data,
    split_practical_data
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
model_count = 500
time_elapsed = 0
start_time = time()

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
    time_taken = time() - start_time - time_elapsed
    # print(f'{Fore.GREEN}  Model {i+1} trained in '
    #       f'{time_taken:.2f} seconds' + Style.RESET_ALL)

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
    precision=None,
    score=None
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
    if score is not None:
        if score < 0.6:
            print(f'{Fore.RED}    Score: '
                  f'{Fore.CYAN}{score:.4f}' + Style.RESET_ALL)
        else:
            print(f'{Fore.GREEN}    Score: '
                  f'{Fore.CYAN}{score:.4f}' + Style.RESET_ALL)


def get_hyperparameters() -> dict:
    hyperparameters = {}
    with open('hyperparameters.json', 'r') as file:
        hyperparameters = json_load(file)
    return hyperparameters


def select_random_hyperparameters():
    decision_tree_parameters = get_hyperparameters().get(
        'DecisionTreeClassifier')
    return {
        'criterion': choice(
            decision_tree_parameters.get('criterion')),
        'splitter': choice(
            decision_tree_parameters.get('splitter')),
        'max_depth': choice(
            decision_tree_parameters.get('max_depth')),
        'min_samples_split': choice(
            decision_tree_parameters.get('min_samples_split')),
        'min_samples_leaf': choice(
            decision_tree_parameters.get('min_samples_leaf')),
        'min_weight_fraction_leaf': choice(
            decision_tree_parameters.get('min_weight_fraction_leaf')),
        'max_features': choice(
            decision_tree_parameters.get('max_features')),
        'random_state': choice(
            decision_tree_parameters.get('random_state')),
        'max_leaf_nodes': choice(
            decision_tree_parameters.get('max_leaf_nodes')),
        'min_impurity_decrease': choice(
            decision_tree_parameters.get('min_impurity_decrease')),
        'class_weight': choice(
            decision_tree_parameters.get('class_weight'))
    }



print(Fore.MAGENTA + f"\nTraining {model_count} models..." + Style.RESET_ALL)


top_models = int(model_count * 0.1)
top_x_models = [
    # {
    #     'score': 0,
    #     'hyperparameters': None,
    #     'f1': 0,
    #     'accuracy': 0,
    #     'recall': 0,
    #     'precision': 0
    # }
]


def get_top_scores():
    return sorted(top_x_models, key=lambda x: x['score'], reverse=True)


def calc_score(f1, accuracy, recall, precision, p_f1, p_accuracy):
    random_weight = randint(1, 50) / 100.0

    # normalize the basic and practical metrics
    basic_metrics = (f1 + accuracy + recall + precision) / 4
    practical_metrics = (p_f1 + p_accuracy) / 2

    # return the weighted average of the basic and practical metrics
    return (
        (random_weight * basic_metrics)
        + ((1 - random_weight) * practical_metrics)
    )


def update_top_scores(
    hyperparameters,
    f1, accuracy, recall, precision,
    p_f1, p_accuracy, p_total_positive_predicted
):
    score = calc_score(f1, accuracy, recall, precision, p_f1, p_accuracy)
    top_model_arr = get_top_scores()
    if len(top_model_arr) < top_models:
        top_model_arr.append({
            'score': score,
            'hyperparameters': hyperparameters,
            'f1': f1,
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'p_f1': p_f1,
            'p_accuracy': p_accuracy,
            'p_total_positive_predicted': p_total_positive_predicted
        })
    else:
        if score > top_model_arr[-1]['score']:
            top_model_arr[-1] = {
                'score': score,
                'hyperparameters': hyperparameters,
                'f1': f1,
                'accuracy': accuracy,
                'recall': recall,
                'precision': precision,
                'p_f1': p_f1,
                'p_accuracy': p_accuracy,
                'p_total_positive_predicted': p_total_positive_predicted
            }
    return top_model_arr


print(f"{Fore.MAGENTA}\nResampling training data...{Style.RESET_ALL}")
x_train_resampled, y_train_resampled = SMOTE().fit_resample(x_train, y_train)
print(f"{Fore.GREEN}Resampling complete{Style.RESET_ALL}")

print(f"{Fore.MAGENTA}\nTraining {model_count} models "
      f"to find best {top_models}...{Style.RESET_ALL}")

for i in tqdm(range(model_count)):
    # print(Fore.MAGENTA + f"\n  Training Model {i + 1}..." + Style.RESET_ALL)
    hyperparameters = select_random_hyperparameters()
    model = None
    accuracy, f1, recall, precision = 0, 0, 0, 0
    p_accuracy, p_f1, p_total_positive_predicted = 0, 0, 0

    model = DecisionTreeClassifier(**hyperparameters)
    model.fit(x_train_resampled, y_train_resampled)

    (
        _, accuracy, f1, recall, precision, time_taken
    ) = evaluate_model(
        model, x_test, y_test, start_time, time_elapsed
    )

    time_elapsed += time_taken

    (
        _, p_accuracy, p_f1, p_total_positive_predicted
    ) = evaluate_practical(
        model, x_practical_test, y_practical_test
    )

    top_x_models = update_top_scores(
        hyperparameters,
        f1, accuracy, recall, precision,
        p_f1, p_accuracy, p_total_positive_predicted
    )


print(Fore.GREEN + f'\nAll models trained and evaluated in '
      f'{Fore.CYAN}{time_elapsed:.2f} seconds' + Style.RESET_ALL)


# =============================================================================
#                           Average metrics
# =============================================================================
print(Fore.RED + "\n--------------------------------" + Style.RESET_ALL)
print(Fore.RED + "        Average Metrics" + Style.RESET_ALL)
print(Fore.RED + "--------------------------------" + Style.RESET_ALL)

avg_score, avg_accuracy, avg_f1, avg_recall, avg_precision = 0, 0, 0, 0, 0
avg_practical_accuracy, avg_practical_f1, avg_positive_predicted = 0, 0, 0

for model in top_x_models:
    avg_score += model['score']
    avg_accuracy += model['accuracy']
    avg_f1 += model['f1']
    avg_recall += model['recall']
    avg_precision += model['precision']

    avg_practical_f1 += model['p_f1']
    avg_practical_accuracy += model['p_accuracy']
    avg_positive_predicted += model['p_total_positive_predicted']

avg_score /= len(top_x_models)
avg_accuracy /= len(top_x_models)
avg_f1 /= len(top_x_models)
avg_recall /= len(top_x_models)
avg_precision /= len(top_x_models)

avg_practical_f1 /= len(top_x_models)
avg_practical_accuracy /= len(top_x_models)
avg_positive_predicted /= len(top_x_models)

print_metrics(
    None,
    avg_accuracy,
    avg_f1,
    avg_recall,
    avg_precision,
    avg_score
)

# =============================================================================
#                           Practical test
# =============================================================================

print(Fore.RED + "\n--------------------------------" + Style.RESET_ALL)
print(f'{Fore.MAGENTA}     Practical Test Results' + Style.RESET_ALL)
print(Fore.RED + "--------------------------------" + Style.RESET_ALL)

total_positive_actual = sum(y_practical_test)

print(f'{Fore.LIGHTYELLOW_EX}  Average Predicted Attendance: '
      f'{Fore.CYAN}{avg_positive_predicted:.1f}' + Style.RESET_ALL)
print(f'{Fore.LIGHTYELLOW_EX}  Actual Attendance: '
      f'{Fore.CYAN}{total_positive_actual}' + Style.RESET_ALL)
print(f'{Fore.LIGHTYELLOW_EX}  Total Students: '
      f'{Fore.CYAN}{len(y_practical_test)}{Style.RESET_ALL}\n')

print_metrics(
    None,
    avg_practical_accuracy,
    avg_practical_f1
)

print()


def calc_best_hyperparameter_range():
    hyperparameters = {}
    for model in top_x_models:
        for key, value in model['hyperparameters'].items():
            if key in hyperparameters:
                hyperparameters[key].append(value)
            else:
                hyperparameters[key] = [value]

    return hyperparameters


def get_best_hyperparameters():
    best_hyperparameters = calc_best_hyperparameter_range()
    for key, value in best_hyperparameters.items():
        best_hyperparameters[key] = max(set(value), key=value.count)

    return best_hyperparameters


# =============================================================================
#                           Top models
# =============================================================================
print(Fore.RED + "\n--------------------------------" + Style.RESET_ALL)
print(f'{Fore.MAGENTA}     Top {top_models} Models' + Style.RESET_ALL)
print(Fore.RED + "--------------------------------" + Style.RESET_ALL)

for i, model in enumerate(top_x_models):
    print(f'{Fore.LIGHTYELLOW_EX}  Model {i + 1} Hyperparameters:'
          f'\n{Fore.CYAN}{model["hyperparameters"]}{Style.RESET_ALL}')
    print_metrics(
        None,
        model['accuracy'],
        model['f1'],
        model['recall'],
        model['precision']
    )
    print(f'{Fore.LIGHTYELLOW_EX}  Practical Test Results' + Style.RESET_ALL)
    print_metrics(
        None,
        model['p_accuracy'],
        model['p_f1']
    )
    print()


# =============================================================================
#                           Best hyperparameters
# =============================================================================

print(Fore.RED + "\n--------------------------------" + Style.RESET_ALL)
print(f'{Fore.MAGENTA}     Best Hyperparameters' + Style.RESET_ALL)
print(Fore.RED + "--------------------------------" + Style.RESET_ALL)

best_hyperparameters = get_best_hyperparameters()
print(f'{Fore.LIGHTYELLOW_EX}  Hyperparameters:'
      f'\n{Fore.CYAN}{best_hyperparameters}{Style.RESET_ALL}')
print()
