import pandas as pd
import joblib
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# IMPORTANT PARAMETERS
TRAIN_CITIES = ["nyc", "chicago", "la", "atlanta", "austin", "sa", "houston", "indy"]
TEST_CITY = "dallas"
FOLDER_PATH = "../data/"

feature_labels = ['Pop Density', 'Intersections', 'Pedways', 'Bikeways', 'POIs', 'Transit']
target_label = "WalkScore"

train_dfs = []
for city in TRAIN_CITIES:
    df = pd.read_csv(f"{FOLDER_PATH}{city}.csv").sample(1200, random_state=42)
    train_dfs.append(df)
train_data = pd.concat(train_dfs, ignore_index=True)
train_data = shuffle(train_data, random_state=42)

test_data = pd.read_csv(f"{FOLDER_PATH}{TEST_CITY}.csv")

X_train = train_data[feature_labels]
y_train = train_data[target_label]
X_test = test_data[feature_labels]
y_test = test_data[target_label]

# Load best parameters and train model
best_params = joblib.load("../models/random_forest_best_params.pkl")
model = RandomForestRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# Error for map
test_data["Error"] = abs(y_test - y_pred)
gdf = gpd.GeoDataFrame(
    test_data,
    geometry=gpd.points_from_xy(test_data["Lon"], test_data["Lat"]),
    crs="EPSG:4326"
    ).to_crs(epsg=3857)

# Plot map
gdf["ErrorSqrt"] = gdf["Error"] ** 0.5

fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, column="ErrorSqrt", cmap="coolwarm", legend=True, markersize=40, alpha=0.8)

# Outline Dallas city boundary
boundary = gpd.read_file("../boundaries/dallas/dallas.shp").to_crs(epsg=3857)
boundary.boundary.plot(ax=ax, edgecolor="black", linewidth=1)

ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

plt.title("Dallas Walk Score Prediction (Absolute Error)")
plt.axis("off")
plt.tight_layout()
plt.savefig("dallas_walkscore_error_map.png", dpi=300)
plt.show()

# Plot feature importance
feature_importance = model.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Random Forest Feature Importance (Dallas)')
plt.show()