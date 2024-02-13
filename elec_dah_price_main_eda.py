# Import the Libraries
import pandas as pd

from utilities import DataLoader, Preprocessor, FeatureEngineering, Plotter


# Load the Data
file_path="./data/elec_dah_price_fr.csv"

dataloader= DataLoader(file_path)
    
df = dataloader.load_elec_dah_data()

print(df.head())

df['Datetime'] = pd.to_datetime(df['Datetime'])

print(df.info())

col_name_scr='France Hr Elec price [€/MWh]'
col_name_tar='hr_elec_price'
# Use the Preprocessor class to rename the column
Preprocessor.rename_col(df, col_name_scr, col_name_tar)

print(df.head())

feateng=FeatureEngineering(df)

feateng.extract_datetime() \
  .add_is_weekend() \
  .add_season() \
  .add_public_holiday(country='France') \
  .add_lag_features([1, 24]) \
  .add_rolling_features([3, 24]) \
  .add_price_change()

df_transformed = feateng.get_features()

Preprocessor.find_null_row(df_transformed, col_name='rolling_std_24h')


print(df_transformed.columns)

Preprocessor.fill_null_values(df_transformed, col_name='price_lag_1h', method='median')
Preprocessor.fill_null_values(df_transformed, col_name='price_lag_24h', method='median')
Preprocessor.fill_null_values(df_transformed, col_name='rolling_mean_3h', method='median')
Preprocessor.fill_null_values(df_transformed, col_name='rolling_std_3h', method='median')
Preprocessor.fill_null_values(df_transformed, col_name='rolling_mean_24h', method='median')
Preprocessor.fill_null_values(df_transformed, col_name='rolling_std_24h', method='median')
Preprocessor.fill_null_values(df_transformed, col_name='price_change_1h', method='median')
Preprocessor.fill_null_values(df_transformed, col_name='price_change_24h', method='median')

print(df_transformed.info())

print(df_transformed.head())

plotter=Plotter(df_transformed)

plotter.plot_timeseries()

plotter.plot_scatterplot()

plotter.plot_boxplot_category()

plotter.plot_correlation_heatmap()

file_name= "./data/df_transformed.csv"

# df_transformed.to_csv(file_name)

