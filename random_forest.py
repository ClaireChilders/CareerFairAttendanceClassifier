# =============================================================================
#                           Load and preprocess data
# =============================================================================
from colorama import Fore, Style
from preprocessing import clean_data, get_practical_test, load_data

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score
)

cleaned_data = clean_data(*load_data())

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
#                           Hyperparameter tuning
# =============================================================================

param_grid = {
    'n_estimators': [500, 1000],
    'max_depth': [None, 1000, 10000],
    'min_samples_split': [10, 15, 20],
    'min_samples_leaf': [4, 8, 12],
}

grid_search = GridSearchCV(RandomForestClassifier(
), param_grid, cv=2, scoring='accuracy', verbose=3, n_jobs=-1)
grid_search.fit(x_train, y_train)

print(grid_search.best_params_)

best_model = RandomForestClassifier(**grid_search.best_params_)
best_model.fit(x_train, y_train)

# =============================================================================
#                           Output results
# =============================================================================

y_pred = best_model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print(Fore.CYAN + "\nResults:" + Style.RESET_ALL)
print(f"  Mean squared error: {mse}")
print(f"  Accuracy: {accuracy}")
print(f"  F1 score: {f1}")
print(f"  Recall: {recall}")
print(f"  Precision: {precision}")

# =============================================================================
#                           Evaluate practical test
# =============================================================================

y_pred = best_model.predict(x_practical_test)
mse = mean_squared_error(y_practical_test, y_pred)
accuracy = accuracy_score(y_practical_test, y_pred)
f1 = f1_score(y_practical_test, y_pred)
recall = recall_score(y_practical_test, y_pred)
precision = precision_score(y_practical_test, y_pred)

positive_predicted = sum(y_pred)

print(Fore.CYAN + "\nPractical test results:" + Style.RESET_ALL)
print(f"  Mean squared error: {mse}")
print(f"  Accuracy: {accuracy}")
print(f"  F1 score: {f1}")
print(f"  Recall: {recall}")
print(f"  Precision: {precision}")
print(f"  Total positive predicted: {positive_predicted}")
