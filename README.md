Brief Overview:
- This project applies a Hybrid Physics ML residual modeling approach to predict concentration output in a Continuous Stirred Tank Reactor (CSTR). The governing physics equation, first order reaction kinetics, provides a baseline prediction using a documented reaction rate constant from research. A machine learning correction layer then learns the systematic gap between the physics prediction and actual sensor readings, accounting for real world deviations the equation cannot capture. Three candidate models were benchmarked on the residual to identify the best performing correction layer.

Model Scope:
- Predictions are grounded in first order reaction kinetics assuming a constant reaction rate k. Real world reactors introduce deviations such as temperature fluctuations, feed flow inconsistencies, and catalyst degradation that a purely physics based model cannot fully account for. The machine learning correction layer was trained to learn these systematic deviations using temperature and feed flow sensor readings alongside elapsed time as input features.

Method:
- Sensor data consisting of 600 readings taken every 20 seconds was loaded and sorted chronologically. A physics baseline was generated using the first order reaction kinetics equation with the researched k value. The residual, defined as actual concentration minus physics prediction, was computed at every timestamp. Three different ML models were tested to see which one best learns the gap between the physics prediction and the actual sensor readings: Linear Regression, Gradient Boosting, and XGBoost. Model selection was based on RMSE and R² performance on the residual.

Correction Layer Model Selection:
- Linear Regression: RMSE 0.0938, MAE 0.0758, R² 0.1402
- Gradient Boosting: RMSE 0.0389, MAE 0.0310, R² 0.8522
- XGBoost: RMSE 0.0073, MAE 0.0053, R² 0.9949
- XGBoost was selected as the correction layer based on superior performance across all metrics. The final hybrid prediction combines the physics baseline with the XGBoost correction. SHAP explainability was applied to the correction layer to identify which features most drove the physics gap.

Final Model Comparison:
- Physics Only: RMSE 0.3277, MAE 0.3119, MAPE 37.99%, R² 0.9352
- Hybrid Model: RMSE 0.0073, MAE 0.0053, MAPE 0.67%, R² 1.0000
- The hybrid model reduced average prediction error from 37.99% to 0.67% by correcting the systematic gap caused by the documented k underestimating the true reaction rate.

SHAP Explainability:
- SHAP analysis on the correction layer identified elapsed time as the dominant driver of the physics gap, followed by feed flow rate and temperature.

Real World Implementation Note:
- In a real world deployment, sensor readings for temperature and feed flow would be fed directly into the correction layer at each prediction step via a live data pipeline. The physics baseline would be recomputed at each interval using current conditions, with the XGBoost correction applied in real time. The model would require periodic retraining as catalyst degrades over time, since degradation shifts the effective k value beyond what the current correction layer was trained on.
- In a production setting, this pipeline would integrate with a SCADA or DCS system via REST API, where live sensor readings are automatically pulled at each prediction interval. The predicted concentration would be written back as a custom tag, enabling operators to monitor forecasted reactor output in real time, set threshold based alerts for concentration drops, and make proactive operational decisions, effectively transforming the control system from a reactive monitoring tool into a predictive decision support system.

Key Insights:
- The hybrid physics ML approach demonstrated that the client's documented reaction rate k significantly underestimates the true effective rate, producing a 37.99% average prediction error when used alone. By combining the physics equation with a data driven correction layer, the hybrid model achieves near perfect prediction accuracy at 0.67% MAPE. Critically, the physics component ensures the model remains physically grounded and interpretable, while the XGBoost correction captures what the equation misses, providing a robust, explainable, and deployable solution for real industrial reactor monitoring. 
