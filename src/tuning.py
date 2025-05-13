import joblib
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


# IMPORTANT PARAMETERS
RANDOM_ITERATIONS = 100 # Amount of parameter combinations to test for each model
TRAIN_CITIES = ["nyc", "chicago", "la", "atlanta", "austin", "sa", "houston", "indy"]
FOLDER_PATH = "../data/"
SAMPLES_PER_CITY = 1200

feature_labels = ['Pop Density', 'Intersections', 'Pedways', 'Bikeways', 'POIs', 'Transit']
target_label = 'WalkScore'


train_dfs = []
for city in TRAIN_CITIES:
    df = pd.read_csv(f"{FOLDER_PATH}{city}.csv").sample(SAMPLES_PER_CITY, random_state=42)
    df["City"] = city
    train_dfs.append(df)
train_data = pd.concat(train_dfs, ignore_index=True)

X_train = train_data[feature_labels].values
y_train = train_data[target_label].values
groups = train_data["City"].values


decision_tree = RandomizedSearchCV(
    estimator=DecisionTreeRegressor(random_state=42),
    param_distributions= {
        "max_depth": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "min_samples_split": randint(2, 20),
    },
    n_iter=RANDOM_ITERATIONS,
    scoring='r2',
    cv=LeaveOneGroupOut().split(X_train, y_train, groups=groups),
    random_state=42,
    n_jobs=-1,
    verbose=1
)
decision_tree.fit(X_train, y_train)
decision_tree_best_params = decision_tree.best_params_
decision_tree_best_score = decision_tree.best_score_


random_forest = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions= {
        "n_estimators": randint(200, 600),
        "max_depth": randint(5, 10),
        "min_samples_leaf": randint(3, 15),
        "min_samples_split": randint(10, 50),
        "max_features": uniform(0.2, 0.8)
    },
    n_iter=RANDOM_ITERATIONS,
    scoring='r2',
    cv=LeaveOneGroupOut().split(X_train, y_train, groups=groups),
    n_jobs=-1,
    verbose=1
)
random_forest.fit(X_train, y_train)
random_forest_best_model = random_forest.best_estimator_
random_forest_best_params = random_forest.best_params_
random_forest_best_score = random_forest.best_score_


xgboost = RandomizedSearchCV(
    estimator=XGBRegressor(objective='reg:squarederror', random_state=42),
    param_distributions= {
        "n_estimators": randint(50, 500),
        "max_depth": randint(2, 10),
        "learning_rate": uniform(0.01, 0.3),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 10),
    },
    n_iter=RANDOM_ITERATIONS,
    scoring='r2',
    cv=LeaveOneGroupOut().split(X_train, y_train, groups=groups),
    n_jobs=-1,
    verbose=1
)
xgboost.fit(X_train, y_train)
xgboost_best_model = xgboost.best_estimator_
xgboost_best_params = xgboost.best_params_
xgboost_best_score = xgboost.best_score_


stack = StackingRegressor(
    estimators= [
        ("rf", random_forest_best_model),
        ("xgb", xgboost_best_model),
    ],
)
stack.fit(X_train, y_train)
stack_scores = cross_val_score(
    stack,
    X_train,
    y_train,
    cv=LeaveOneGroupOut().split(X_train, y_train, groups=groups),
    scoring=make_scorer(r2_score),
    n_jobs=-1
)
stack_best_params = stack.get_params(deep=True)


print("Decision Tree Best params:", decision_tree_best_params)
print("Decision Tree Best validation score:", decision_tree_best_score)
print("Random Forest Best params:", random_forest_best_params)
print("Random Forest Best validation score:", random_forest_best_score)
print("XGBoost Best params:", xgboost_best_params)
print("XGBoost Best validation score:", xgboost_best_score)
print("Stacking validation score:", stack_scores.mean())

joblib.dump(decision_tree_best_params, '../models/decision_tree_best_params.pkl')
joblib.dump(random_forest_best_params, '../models/random_forest_best_params.pkl')
joblib.dump(xgboost_best_params, '../models/xgboost_best_params.pkl')
joblib.dump(stack_best_params, '../models/stacking_best_params.pkl')
