import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

from utils import DataLoader, Preprocessor, Visualization, GrangerCausality

# Load the Data

file_path="./data/electricity_dah_prices.csv"

dataloader = DataLoader(file_path)

df = dataloader.load_elec_dah_data_index()

print(df.head())

print(df.shape)

print(df.info())

print(df.columns)

print(df.describe())

countries = ['france', 'italy', 'belgium', 'spain', 'uk', 'germany']

vis=Visualization(df)

vis.plot_elec_dah_price_countries(countries)

vis.plot_correlation_matrix(countries)


prep=Preprocessor

col_to_drop =["uk", "italy", "belgium"]

df1=prep.drop_columns(df, col_to_drop).copy()

print(df1.head())


# To find the null value row
prep.find_null_row(df1, col_name='spain')

# Check NaN status
f = df1[df1['hour'] == '02:00 - 03:00'] 

print(f.describe().T)


# Boxplot shows presence of outliers 

vis.plot_boxplot(df, 'france')

vis.plot_boxplot(df, 'germany')

vis.plot_boxplot(df, 'spain')


prep.fill_null_values(df1, 'france', method='mean', condition=prep.filter_hour_02_to_03)
prep.fill_null_values(df1, 'germany', method='median', condition=prep.filter_hour_02_to_03)
prep.fill_null_values(df1, 'spain', method='median', condition=prep.filter_hour_02_to_03)

print(df1.info())

gc=GrangerCausality(df1)
 
gc.granger_causality_analysis('france')

# Renaming columns to include the unit
col_name_scr = {'france', 'spain', 'germany'}
col_name_tar={'France Hr Elec price [€/MWh]', 'Spain Hr Elec price [€/MWh]', 'Germany Hr Elec price [€/MWh]'}
prep.rename_columns(df1, col_name_scr, col_name_tar)

print(df1.head())

df1 = df1.reset_index('date')

prep.convert_datetime(df1)

print(df1.head())

prep.extract_hr(df1)

print(df1.head())

col_to_drop1 =["date", "hour", "Hr", "Germany Hr Elec price [€/MWh]", "Spain Hr Elec price [€/MWh]"]

df1=prep.drop_columns(df1, col_to_drop1)

print(df1.head())

df1= prep.remove_outliers(df1)

print(df1.shape)

df1 = df1.set_index('Datetime')

print(df1.info())

df1.index=pd.to_datetime(df1.index)

print(df1.head())

vis.plot_elec_dah_price(df1, col_name='France Hr Elec price [€/MWh]')

file_name= "./data/elec_dah_price_fr.csv"

# df1.to_csv(file_name)


