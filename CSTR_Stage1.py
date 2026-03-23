import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
import joblib
import json

df = pd.read_csv('reactor_data.csv')
print(df)

df = df.sort_values('timestamp').reset_index(drop=True)

df['seconds'] = (pd.to_datetime(df['timestamp']) - pd.to_datetime(df['timestamp'].iloc[0])).dt.total_seconds()

k_assumed = 0.0002

C0 = df['concentration'].iloc[0]

df['C_physics'] = C0 * np.exp(-k_assumed * df['seconds'])

# Visual check — physics curve must follow same trend as actual
plt.plot(df['seconds'], df['concentration'], label='Actual')
plt.plot(df['seconds'], df['C_physics'], label='Physics')
plt.legend()
plt.title('Actual vs Physics')
plt.show()

# Residual plotting 
df['residual'] = df['concentration'] - df['C_physics']

plt.figure(figsize=(12, 5))
plt.plot(df['seconds'], df['concentration'], label='Actual', color='blue')
plt.plot(df['seconds'], df['C_physics'], label='Physics', color='orange')
plt.plot(df['seconds'], df['residual'], label='Residual', color='green')
plt.axhline(0, color='red', linestyle='--')
plt.title('Residual over time')
plt.legend()
plt.show()

# Defining parameters
X = df[['temperature', 'feed_flow', 'seconds']].values
y = df['residual'].values
C_physics_all = df['C_physics'].values
C_actual_all = df['concentration'].values

# TimeSeriesSplit cross validation
tscv = TimeSeriesSplit(n_splits=5)

def get_model(name):
    if name == 'Linear Regression':
        return LinearRegression()
    elif name == 'Gradient Boosting':
        return GradientBoostingRegressor()
    else:
        return xgb.XGBRegressor()

model_names = ['Linear Regression', 'Gradient Boosting', 'XGBoost']

# Store CV scores per model for overfitting check
cv_scores = {}

print("\n--- RESIDUAL MODEL BENCHMARKING (TimeSeriesSplit CV) ---")

best_model_name = None
best_rmse = float('inf')

for name in model_names:
    rmse_scores, mae_scores, r2_scores = [], [], []

    for train_idx, test_idx in tscv.split(X):
        model = get_model(name)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])

        rmse_scores.append(np.sqrt(mean_squared_error(y[test_idx], preds)))
        mae_scores.append(mean_absolute_error(y[test_idx], preds))
        r2_scores.append(r2_score(y[test_idx], preds))

    avg_rmse = np.mean(rmse_scores)

    # Train on full dataset for overfitting check
    full_model = get_model(name)
    full_model.fit(X, y)
    train_preds = full_model.predict(X)
    train_rmse = np.sqrt(mean_squared_error(y, train_preds))
    train_mae = mean_absolute_error(y, train_preds)
    train_r2 = r2_score(y, train_preds)

    cv_scores[name] = {
        'cv_rmse': avg_rmse,
        'cv_mae': np.mean(mae_scores),
        'cv_r2': np.mean(r2_scores),
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2
    }

    print(f"\n{name}")
    print(f"  Train RMSE: {train_rmse:.4f}   CV RMSE: {avg_rmse:.4f} ± {np.std(rmse_scores):.4f}   RMSE Gap: {train_rmse - avg_rmse:.4f}")
    print(f"  Train MAE:  {train_mae:.4f}   CV MAE:  {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}   MAE Gap: {train_mae - np.mean(mae_scores):.4f}")
    print(f"  Train R²:   {train_r2:.4f}   CV R²:   {np.mean(r2_scores):.4f}   R² Gap: {train_r2 - np.mean(r2_scores):.4f}")

    # R² std excluded — large negative values in early folds make std misleading
    if avg_rmse < best_rmse:
        best_rmse = avg_rmse
        best_model_name = name

print(f"\nBest model: {best_model_name} (CV RMSE: {best_rmse:.4f})")

# Train final model on full dataset using best model from CV
ml_model = get_model(best_model_name)
ml_model.fit(X, y)

# HYBRID PREDICTION
df['ml_correction'] = ml_model.predict(X)

df['C_hybrid'] = df['C_physics'] + df['ml_correction']

# PHYSICS vs HYBRID 
print("\n--- PHYSICS vs HYBRID COMPARISON (TimeSeriesSplit CV) ---")
physics_rmse, physics_mae, physics_r2 = [], [], []
hybrid_rmse, hybrid_mae, hybrid_r2 = [], [], []

for train_idx, test_idx in tscv.split(X):
    fold_model = get_model(best_model_name)
    fold_model.fit(X[train_idx], y[train_idx])

    fold_correction = fold_model.predict(X[test_idx])
    fold_physics = C_physics_all[test_idx]
    fold_hybrid = fold_physics + fold_correction
    fold_actual = C_actual_all[test_idx]

    physics_rmse.append(np.sqrt(mean_squared_error(fold_actual, fold_physics)))
    physics_mae.append(mean_absolute_error(fold_actual, fold_physics))
    physics_r2.append(r2_score(fold_actual, fold_physics))

    hybrid_rmse.append(np.sqrt(mean_squared_error(fold_actual, fold_hybrid)))
    hybrid_mae.append(mean_absolute_error(fold_actual, fold_hybrid))
    hybrid_r2.append(r2_score(fold_actual, fold_hybrid))

# Physics train metrics
physics_train_rmse = np.sqrt(mean_squared_error(C_actual_all, C_physics_all))
physics_train_mae = mean_absolute_error(C_actual_all, C_physics_all)
physics_train_r2 = r2_score(C_actual_all, C_physics_all)

# Hybrid train metrics
hybrid_train_preds = df['C_hybrid'].values
hybrid_train_rmse = np.sqrt(mean_squared_error(C_actual_all, hybrid_train_preds))
hybrid_train_mae = mean_absolute_error(C_actual_all, hybrid_train_preds)
hybrid_train_r2 = r2_score(C_actual_all, hybrid_train_preds)

cv_physics_rmse = np.mean(physics_rmse)
cv_hybrid_rmse = np.mean(hybrid_rmse)

print(f"\nPhysics Only")
print(f"  Train RMSE: {physics_train_rmse:.4f}   CV RMSE: {cv_physics_rmse:.4f} ± {np.std(physics_rmse):.4f}   RMSE Gap: {physics_train_rmse - cv_physics_rmse:.4f}")
print(f"  Train MAE:  {physics_train_mae:.4f}   CV MAE:  {np.mean(physics_mae):.4f} ± {np.std(physics_mae):.4f}   MAE Gap: {physics_train_mae - np.mean(physics_mae):.4f}")
print(f"  Train R²:   {physics_train_r2:.4f}   CV R²:   {np.mean(physics_r2):.4f}   R² Gap: {physics_train_r2 - np.mean(physics_r2):.4f}")

print(f"\nHybrid Model (Physics + ML Correction) — {best_model_name}")
print(f"  Train RMSE: {hybrid_train_rmse:.4f}   CV RMSE: {cv_hybrid_rmse:.4f} ± {np.std(hybrid_rmse):.4f}   RMSE Gap: {hybrid_train_rmse - cv_hybrid_rmse:.4f}")
print(f"  Train MAE:  {hybrid_train_mae:.4f}   CV MAE:  {np.mean(hybrid_mae):.4f} ± {np.std(hybrid_mae):.4f}   MAE Gap: {hybrid_train_mae - np.mean(hybrid_mae):.4f}")
print(f"  Train R²:   {hybrid_train_r2:.4f}   CV R²:   {np.mean(hybrid_r2):.4f}   R² Gap: {hybrid_train_r2 - np.mean(hybrid_r2):.4f}")

# FINAL PLOT
plt.figure(figsize=(14, 6))
plt.plot(df['seconds'], df['concentration'], label='Actual', color='blue', alpha=0.6)
plt.plot(df['seconds'], df['C_physics'], label='Physics Only', color='orange', linewidth=2)
plt.plot(df['seconds'], df['C_hybrid'], label=f'Hybrid Model — {best_model_name}', color='green', linewidth=2)
plt.title('Hybrid Physics-ML Model vs Physics Only vs Actual')
plt.xlabel('Time (seconds)')
plt.ylabel('Concentration')
plt.legend()
plt.tight_layout()
plt.show()

# SHAP EXPLAINABILITY
# Automatically switches explainer based on winning model
X_df = df[['temperature', 'feed_flow', 'seconds']]
if best_model_name == 'Linear Regression':
    explainer = shap.LinearExplainer(ml_model, X_df)
else:
    explainer = shap.TreeExplainer(ml_model)

shap_values = explainer.shap_values(X_df)

plt.figure()
shap.summary_plot(shap_values, X_df, feature_names=['temperature', 'feed_flow', 'seconds'], show=True)