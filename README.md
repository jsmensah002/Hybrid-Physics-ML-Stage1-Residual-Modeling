Brief Overview:
- This project applies a Hybrid Physics ML residual modeling approach to predict concentration output in a Continuous Stirred Tank Reactor (CSTR). The governing physics equation, first order reaction kinetics, provides a baseline prediction using a documented reaction rate constant from research. A machine learning correction layer then learns the systematic gap between the physics prediction and actual sensor readings, accounting for real world deviations the equation cannot capture. Three candidate models were benchmarked on the residual to identify the best performing correction layer.

Model Scope:
- Predictions are grounded in first order reaction kinetics assuming a constant reaction rate k. Real world reactors introduce deviations such as temperature fluctuations, feed flow inconsistencies, and catalyst degradation that a purely physics based model cannot fully account for. The machine learning correction layer was trained to learn these systematic deviations using temperature and feed flow sensor readings alongside elapsed time as input features.

Method:
- Sensor data consisting of 600 readings taken every 20 seconds was loaded and sorted chronologically. A physics baseline was generated using the first order reaction kinetics equation with the researched k value. The residual, defined as actual concentration minus physics prediction, was computed at every timestamp. Three different ML models were tested to see which one best learns the gap between the physics prediction and the actual sensor readings: Linear Regression, Gradient Boosting, and XGBoost. Model selection was based on CV RMSE and R² performance on the residual.

Model Selection (CV means Cross Validation):
- Linear Regression: Train RMSE: 0.0938 | CV RMSE: 0.1369 ± 0.0825 | RMSE Gap: -0.0431| Train MAE: 0.0758 | CV MAE: 0.1187 ± 0.0789 | MAE Gap: -0.0429| Train R²: 0.1402 | CV R²: -9.5906 | R² Gap: 9.7309
- Gradient Boosting: Train RMSE: 0.0389 | CV RMSE: 0.0744 ± 0.0272 | RMSE Gap: -0.0355 | Train MAE: 0.0310 | CV MAE: 0.0628 ± 0.0257 | MAE Gap: -0.0318 | Train R²: 0.8522 | CV R²: -1.2784 | R² Gap: 2.1306
- XGBoost: Train RMSE: 0.0073 | CV RMSE: 0.0887 ± 0.0373 | RMSE Gap: -0.0814 | Train MAE: 0.0053 | CV MAE: 0.0758 ± 0.0351 | MAE Gap: -0.0705 | Train R²: 0.9949 | CV R²: -2.3904 | R² Gap: 3.3853
- Across all metrics, Gradient Boosting was selected as the best model because it has the lowest CV RMSE of 0.0744. Meaning it makes the smallest average prediction error on unseen data across all 5 folds.

Final Model Comparison:
- Physics Only: Train RMSE: 0.3277 | CV RMSE: 0.3403 ± 0.0617 | RMSE Gap: -0.0125 | Train MAE: 0.3119 | CV MAE: 0.3363 ± 0.0622 | MAE Gap: -0.0243 | Train R²: 0.9352 | CV R²: -5.7717 | R² Gap: 6.7069
- Gradient Boosting Hybrid Model (Physics + ML corrections): Train RMSE: 0.0389 | CV RMSE: 0.0762 ± 0.0283 | RMSE Gap: -0.0373 | Train MAE: 0.0310 | CV MAE: 0.0647 ± 0.0267 | MAE Gap: -0.0337 | Train R²: 0.9991 | CV R²: 0.6188 | R² Gap: 0.3803
- Gradient Boosting Hybrid Model was selected over physics alone because it achieves a CV RMSE of 0.0762 compared to the Physics Only Model at 0.3403, a 78% reduction in prediction error on unseen data and an R² of 0.6188 confirming the model genuinely captures the concentration curve. 

Visual Fit:
- The final plot demonstrates strong alignment between the hybrid model predictions and actual sensor readings across the full 12,000 second operating window, consistent with the CV R² of 0.62 achieved on unseen data.

SHAP Explainability:
- SHAP analysis on the correction layer identified elapsed time as the dominant driver of the physics gap, followed by feed flow rate and temperature.

Real World Implementation Note:
- In a real world deployment, temperature and feed flow sensor readings would be fed into the correction layer at each prediction step via a live data pipeline. The physics baseline would be recomputed at each interval using current conditions with the Gradient Boosting correction applied in real time. The model would require periodic retraining as catalyst degrades over time since degradation shifts the effective k value beyond what the current correction layer was trained on.
- In a production setting this pipeline would integrate with a SCADA or DCS system where live sensor readings are automatically pulled at each prediction interval and the predicted concentration written back as a custom tag, enabling operators to monitor forecasted reactor output in real time, set threshold based alerts for concentration drops, and make proactive operational decisions.

Key Insights:
- The results demonstrate that purely physics based models are insufficient for real industrial reactor monitoring where operating conditions deviate from idealised assumptions. The hybrid approach bridges this gap by combining physical interpretability with data driven adaptability, a methodology directly applicable to any industrial system where a governing equation exists but real world deviations are significant.
