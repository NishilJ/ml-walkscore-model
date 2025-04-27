import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

training_cities = ["atlanta", "chicago", "houston", "la", "nyc"]
training_dataframes = []

for city in training_cities:
    file_path = os.path.join("data", f"{city}.csv")
    dataframe = pd.read_csv(file_path)
    training_dataframes.append(dataframe)

train_data = pd.concat(training_dataframes)

test_data = pd.read_csv(os.path.join("data", "dallas.csv"))

feature_labels = ['Pop Density', 'Intersections', 'Pedways', 'Bikeways', 'POIs', 'Transit']
target_label = 'WalkScore'

X_train = train_data[feature_labels]
y_train = train_data[target_label]

X_test = test_data[feature_labels]
y_test = test_data[target_label]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, 'scaler.pkl')

# training the RandomForestRegressor model
model_rfr = RandomForestRegressor(
    n_estimators=236,
    max_depth=4,
    min_samples_leaf=5,
    min_samples_split=23,
    max_features='sqrt',
    random_state=42,
)
model_rfr.fit(X_train, y_train)

# test prediction
y_pred = model_rfr.predict(X_test)

mae_rfr = mean_absolute_error(y_test, y_pred)
mse_rfr = mean_squared_error(y_test, y_pred)
r2_rfr = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae_rfr}")
print(f"Mean Squared Error (MSE): {mse_rfr}")
print(f"R2 Score: {r2_rfr}\n")

feature_importances = model_rfr.feature_importances_
print("Feature Importances:")
for feature, importance in zip(feature_labels, feature_importances):
    print(f"{feature}: {importance:.4f}")

joblib.dump(model_rfr, 'RandomForestRegressorModel.pkl')
print("Model saved as RandomForestRegressorModel.pkl\n")

print(50*"-")

# training the DecisionTreeRegressor
model_dtr = DecisionTreeRegressor(
    max_depth=5,
    min_samples_leaf=5,
    min_samples_split=10,
    max_features='sqrt',
    random_state=42
)

model_dtr.fit(X_train, y_train)

# test prediction
y_pred = model_dtr.predict(X_test)

mae_dtr = mean_absolute_error(y_test, y_pred)
mse_dtr = mean_squared_error(y_test, y_pred)
r2_dtr = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae_dtr}")
print(f"Mean Squared Error (MSE): {mse_dtr}")
print(f"R2 Score: {r2_dtr}\n")

feature_importances_dtr = model_dtr.feature_importances_
print("Feature Importances:")
for feature, importance in zip(feature_labels, feature_importances_dtr):
    print(f"{feature}: {importance:.4f}")

joblib.dump(model_dtr, 'DecisionTreeRegressorModel.pkl')
print("Model saved as DecisionTreeRegressorModel.pkl")