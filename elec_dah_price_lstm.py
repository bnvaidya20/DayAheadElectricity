# Import the Libraries
import pandas as pd

from keras.models import load_model

from utils import DataLoader, Plotter, DataPreparation, LSTMModel, ModelEvaluator


# Load the Data
file_path="./data/df_transformed.csv"

dataloader= DataLoader(file_path)
    
df_transformed = dataloader.load_elec_dah_data()

print(df_transformed.head())

print(df_transformed.info())

plotter=Plotter(df_transformed)

dataprep=DataPreparation(df_transformed)

# LSTM Data Prep

target_col = 'hr_elec_price'
n_input_steps = 24  # Number of time steps (hours) in each input sequence
n_features = 3  # Number of features to use in the model

X_train, X_test, y_train, y_test, scaler, test_dates =dataprep.prepare_data_lstm(target_col, n_input_steps, n_features)

print(X_train.shape)
print(y_train.shape)

# LSTM Modeling

lstm=LSTMModel(n_input_steps, n_features)

lstm.build_model()

n_epochs=100
batch_size=64

# lstm.train(X_train, y_train, epochs=n_epochs, batch_size=batch_size)

lstm.load_trained_model()

y_pred_ls= lstm.predict(X_test)

eval_ls = ModelEvaluator(y_test, y_pred_ls, "LSTM")

eval_ls.evaluate()

metrics = eval_ls.get_metrics()

plotter.plot_predict_actual(test_dates, y_test, y_pred_ls, "LSTM")






