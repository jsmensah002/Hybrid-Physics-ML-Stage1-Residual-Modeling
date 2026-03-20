import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb

df = pd.read_csv('reactor_data.csv')
print(df)

# Sort by timestamp
df = df.sort_values('timestamp').reset_index(drop=True)

# Convert timestamp to seconds from start
# If system is recorded in mins, hours, days, divide the dt.total_seconds()/ by either 60secs, 3600, or 86400 respectively
df['seconds'] = (pd.to_datetime(df['timestamp']) - pd.to_datetime(df['timestamp'].iloc[0])).dt.total_seconds()

k_assumed = 0.0002   # estimated from equipment specs or research papers

C0 = df['concentration'].iloc[0]

df['C_physics'] = C0 * np.exp(-k_assumed * df['seconds'])

import matplotlib.pyplot as plt
plt.plot(df['seconds'], df['concentration'], label='Actual')
plt.plot(df['seconds'], df['C_physics'], label='Physics')
plt.legend()
plt.title('Actual vs Physics')
plt.show()

# COMPUTE RESIDUAL 
df['residual'] = df['concentration'] - df['C_physics']

# Plot the residual column and you'll see whether it's random or patterned
# either way proceed with ML and let the metrics determine whether ML is needed or not

plt.figure(figsize=(12, 5))
plt.plot(df['seconds'], df['concentration'], label='Actual', color='blue')
plt.plot(df['seconds'], df['C_physics'], label='Physics', color='orange')
plt.plot(df['seconds'], df['residual'], color='green', label='Residual')
plt.axhline(0, color='red', linestyle='--')
plt.title('Residual over time')
plt.legend()
plt.show()

# TRAIN ML ON RESIDUAL 
# X = sensor columns that explain WHY the physics was wrong (domain knowledge + client engineers)
# X is NOT necessarily what the physics equation is a function of
# y = residual (always)

X = df[['temperature', 'feed_flow', 'seconds']]  
y = df['residual']  

# BENCHMARK MODELS ON RESIDUAL
# Same evaluation applied to all three. Let metrics decide the best correction model

models = {
    'Linear Regression': LinearRegression(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'XGBoost': xgb.XGBRegressor()
}

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate(name, actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = r2_score(actual, predicted)
    print(f"\n{name}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  MAPE: {mape:.4f}%")
    print(f"  R²:   {r2:.4f}")

print("\n--- RESIDUAL MODEL BENCHMARKING ---")
for name, model in models.items():
    model.fit(X, y)
    preds = model.predict(X)
    evaluate(name, y, preds)

# Pick best model based on metrics. Replace the model if another model wins
ml_model = xgb.XGBRegressor()
ml_model.fit(X, y)

# HYBRID PREDICTION
# ml_correction: what the ML model predicts the physics got wrong at every row. This is the learned gap.
# T_hybrid: physics prediction plus ML correction. This is your final hybrid model output.

df['ml_correction'] = ml_model.predict(X)           # This is ML's prediction of the residual at every row
df['C_hybrid'] = df['C_physics'] + df['ml_correction']

# EVALUATION 
print("\n--- PHYSICS vs HYBRID COMPARISON ---")

evaluate("Physics Only", df['concentration'], df['C_physics'])
evaluate("Hybrid Model", df['concentration'], df['C_hybrid'])

# FINAL PLOT
plt.figure(figsize=(14, 6))
plt.plot(df['seconds'], df['concentration'], label='Actual', color='blue', alpha=0.6)
plt.plot(df['seconds'], df['C_physics'], label='Physics Only', color='orange', linewidth=2)
plt.plot(df['seconds'], df['C_hybrid'], label='Hybrid Model', color='green', linewidth=2)
plt.title('Hybrid Physics-ML Model vs Physics Only vs Actual')
plt.xlabel('Time (seconds)')
plt.ylabel('Concentration')
plt.legend()
plt.tight_layout()
plt.show()

# SHAP EXPLAINABILITY ON ML CORRECTION LAYER
# SHAP explains which features are driving the physics gap correction

import shap
# Change the Explainer for whichever model picked
# shap.LinearExplainer(ml_model, X) model + data together
# shap.TreeExplainer(ml_model) — model only

explainer = shap.TreeExplainer(ml_model)
shap_values = explainer.shap_values(X)

# Summary plot - which features matter most for correcting the physics gap
plt.figure()
shap.summary_plot(shap_values, X, feature_names=['temperature', 'feed_flow', 'seconds'], show=True)

# Saving the model
import joblib
import json

# Save ML correction model
joblib.dump(ml_model, 'xgboost_correction.pkl')

# Save physics parameters
physics_params = {
    'k': k_assumed,
    'C0': float(C0)
}

with open('physics_params.json', 'w') as f:
    json.dump(physics_params, f)

print("Model saved successfully")