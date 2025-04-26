import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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

# training the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# test prediction
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R2 Score: {r2}\n")

feature_importances = model.feature_importances_
print("Feature Importances:")
for feature, importance in zip(feature_labels, feature_importances):
    print(f"{feature}: {importance:.4f}")

joblib.dump(model, 'RandomForestRegressorModel.pkl')
print("Model saved as RandomForestRegressorModel.pkl")