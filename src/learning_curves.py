import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from itertools import combinations
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# IMPORTANT PARAMETERS
TRAIN_CITIES = ["nyc", "chicago", "la", "atlanta", "austin", "sa", "houston", "indy"]
FOLDER_PATH = "../data/"
SAMPLES_PER_CITY = 1200

# Models and parameter paths
models = {
    "Decision Tree": (DecisionTreeRegressor, "../models/decision_tree_best_params.pkl"),
    "Random Forest": (RandomForestRegressor, "../models/random_forest_best_params.pkl"),
    "XGBoost": (XGBRegressor, "../models/xgboost_best_params.pkl"),
    "Stacking": (StackingRegressor, "../models/stacking_best_params.pkl"),
}
trained_models = {}


feature_labels = ['Pop Density', 'Intersections', 'Pedways', 'Bikeways', 'POIs', 'Transit']
target_label = 'WalkScore'


train_dfs = {}
for city in TRAIN_CITIES:
    df = pd.read_csv(f"{FOLDER_PATH}{city}.csv")
    df["City"] = city
    if len(df) < SAMPLES_PER_CITY:
        raise ValueError(f"City '{city}' has fewer than {SAMPLES_PER_CITY} samples.")
    train_dfs[city] = df.sample(SAMPLES_PER_CITY, random_state=42)

# Run evaluation
for model_name, (model_class, param_file) in models.items():
    print(f"\n===== Evaluating: {model_name} =====")
    try:
        params = joblib.load(param_file)
    except Exception as e:
        print(f"Failed to load parameters for {model_name}: {e}")
        continue

    results = []

    for held_out_city in TRAIN_CITIES:
        other_cities = [c for c in TRAIN_CITIES if c != held_out_city]

        for k in range(1, len(other_cities) + 1):
            for train_cities in combinations(other_cities, k):
                train_data = pd.concat([train_dfs[c] for c in train_cities], ignore_index=True)
                val_data = train_dfs[held_out_city]

                X_train = train_data[feature_labels].values
                y_train = train_data[target_label].values
                X_val = val_data[feature_labels].values
                y_val = val_data[target_label].values

                if model_name == "Stacking":
                    rf_model = trained_models.get("Random Forest")
                    xgb_model = trained_models.get("XGBoost")

                    base_learners = [
                        ("rf", rf_model),
                        ("xgb", xgb_model)
                    ]

                    model = StackingRegressor(
                        estimators=base_learners,
                        n_jobs=-1
                    )
                else:
                    model = model_class(**params)

                model.fit(X_train, y_train)

                # Store trained base models to use in stacking
                if model_name in ["Random Forest", "XGBoost"]:
                    trained_models[model_name] = model

                y_pred = model.predict(X_val)
                y_train_pred = model.predict(X_train)

                results.append({
                    "model": model_name,
                    "held_out_city": held_out_city,
                    "num_train_cities": len(train_cities),
                    "train_r2": r2_score(y_train, y_train_pred),
                    "val_r2": r2_score(y_val, y_pred),
                    "val_mae": mean_absolute_error(y_val, y_pred),
                    "val_rmse": root_mean_squared_error(y_val, y_pred)
                })

    df_results = pd.DataFrame(results)

    # Print average metrics
    avg_r2 = df_results["val_r2"].mean()
    avg_mae = df_results["val_mae"].mean()
    avg_rmse = df_results["val_rmse"].mean()
    print(f"\n{model_name} Summary:")
    print(f"  Avg Validation R²   : {avg_r2:.4f}")
    print(f"  Avg Validation MAE  : {avg_mae:.4f}")
    print(f"  Avg Validation RMSE : {avg_rmse:.4f}")


    # Aggregate results
    agg = df_results.groupby("num_train_cities").agg(
        val_r2_mean=("val_r2", "mean"),
        val_r2_std=("val_r2", "std"),
        train_r2_mean=("train_r2", "mean"),
        train_r2_std=("train_r2", "std"),
        val_mae_mean=("val_mae", "mean"),
        val_mae_std=("val_mae", "std"),
    )

    # Plot R^2
    plt.figure(figsize=(10, 6))
    for label in ["train_r2", "val_r2"]:
        plt.plot(agg.index, agg[f"{label}_mean"], marker='o', label=f"{label.replace('_', ' ').title()}")
        plt.fill_between(
            agg.index,
            agg[f"{label}_mean"] - agg[f"{label}_std"],
            agg[f"{label}_mean"] + agg[f"{label}_std"],
            alpha=0.2
        )

    plt.title(f"{model_name} R² vs. Num of Training Cities")
    plt.xlabel("Num of Training Cities")
    plt.ylabel("R² Score")
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../learning_curves/{model_name.replace(' ', '_').lower()}_r2_learning_curve.png")
    plt.show()

    # Plot MAE
    plt.figure(figsize=(10, 6))
    plt.plot(agg.index, agg["val_mae_mean"], marker='o', label="Validation MAE", color='orange')
    plt.fill_between(
        agg.index,
        agg["val_mae_mean"] - agg["val_mae_std"],
        agg["val_mae_mean"] + agg["val_mae_std"],
        alpha=0.2
    )

    plt.title(f"{model_name} MAE vs. Num of Training Cities")
    plt.xlabel("Num of Training Cities")
    plt.ylabel("Mean Absolute Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../learning_curves/{model_name.replace(' ', '_').lower()}_mae_learning_curve.png")
    plt.show()