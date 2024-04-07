# =============================================================================
#                           Load and preprocess data
# =============================================================================
import sys
import time
from colorama import Fore, Style
from tqdm import tqdm
from preprocessing import (extract_features_target,
                           get_practical_test, load_data, print_metrics)

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score
)

cleaned_data = load_data()

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
#                           Hyperparameter Tuning
# =============================================================================

param_grid = {
    'n_estimators': [4],
    'max_depth': [None],
    'min_samples_split': [8],
    'min_samples_leaf': [1]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='f1',
    verbose=3,
    n_jobs=-1,
    pre_dispatch='n_jobs/2'
)

cleaned_data.drop(
    columns=['career_fair_name'],
    axis=1,
    inplace=True
)
features, target = extract_features_target(cleaned_data)
x_train, x_val, y_train, y_val = train_test_split(
    features, target, test_size=0.2, random_state=42
)

grid_search.fit(x_train, y_train)

print(grid_search.best_params_)

best_model = RandomForestClassifier(**grid_search.best_params_)
best_model.fit(x_train, y_train)

# Evalute on validation data

print(Fore.CYAN + "\nValidation results:" + Style.RESET_ALL)

y_pred = best_model.predict(x_val)
mse = mean_squared_error(y_val, y_pred)
accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)

positive_predicted = sum(y_pred)

print_metrics(mse, accuracy, f1, recall, precision)
print(f"  Total positive predicted: {positive_predicted}")

# Evaluate on practical data

print(Fore.CYAN + "\nPractical test results:" + Style.RESET_ALL)

y_pred = best_model.predict(x_practical_test)
mse = mean_squared_error(y_practical_test, y_pred)
accuracy = accuracy_score(y_practical_test, y_pred)
f1 = f1_score(y_practical_test, y_pred)
recall = recall_score(y_practical_test, y_pred)
precision = precision_score(y_practical_test, y_pred)

positive_predicted = sum(y_pred)

print_metrics(mse, accuracy, f1, recall, precision)

print(f"  Total positive predicted: {positive_predicted}")

# Print the feature ranking

importances = best_model.feature_importances_
indices = importances.argsort()[::-1]

print(Fore.CYAN + "\nFeature importance ranking:" + Style.RESET_ALL)
num_features = x_train.shape[1]
print(f'{Fore.LIGHTBLACK_EX}{"Rank": >4} {"Feature": >40} {"Importance": >10}')
for f in range(x_train.shape[1]):
    if f < 0.15*num_features:
        val_color = Fore.GREEN
    elif f < 0.50*num_features:
        val_color = Fore.CYAN
    elif f < 0.85*num_features:
        val_color = Fore.YELLOW
    else:
        val_color = Fore.RED

    if importances[indices[f]] < 0.00001:
        name_color = Fore.LIGHTBLACK_EX
        val_color = Fore.LIGHTBLACK_EX
    else:
        name_color = Fore.MAGENTA

    print(f'{Fore.LIGHTBLACK_EX}{f+1: >4}'
          f'{name_color}{features.columns[indices[f]]: >40} '
          f'{val_color}{importances[indices[f]]: >10.5f}'
          f'{Style.RESET_ALL}')