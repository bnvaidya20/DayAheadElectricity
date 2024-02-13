# Import the Libraries
import pandas as pd

from utilities import DataLoader, Plotter, DataPreparation, ModelEvaluator, RandomForestModel, XGBoostModel


# Load the Data
file_path="./data/df_transformed.csv"

dataloader= DataLoader(file_path)
    
df_transformed = dataloader.load_elec_dah_data()

print(df_transformed.head())

print(df_transformed.info())

plotter=Plotter(df_transformed)

dataprep=DataPreparation(df_transformed)


# RF/XGB Data Prep
target_col = 'hr_elec_price'

X_train, X_test, y_train, y_test, scaler, test_dates= dataprep.prepare_data_rf_xgb(target_col)

print(X_train.shape)
print(y_train.shape)

# RF Modeling

rf_model= RandomForestModel(n_estimators=100)

rf_model.train(X_train, y_train)

y_pred_rf=rf_model.predict(X_test)


eval_rf = ModelEvaluator(y_test, y_pred_rf, "RF")

eval_rf.evaluate()

metrics = eval_rf.get_metrics()

plotter.plot_predict_actual(test_dates, y_test, y_pred_rf, "RF")

# XGBoost Modeling

xgb_model= XGBoostModel(n_estimators=100)

xgb_model.train(X_train, y_train)

y_pred_xgb=xgb_model.predict(X_test)


eval_xgb = ModelEvaluator(y_test, y_pred_xgb, "XGB")

eval_xgb.evaluate()

metrics = eval_xgb.get_metrics()

plotter.plot_predict_actual(test_dates, y_test, y_pred_xgb, "XGB")

