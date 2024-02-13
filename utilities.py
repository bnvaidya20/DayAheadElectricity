import pandas as pd
import numpy as np
from scipy import stats
import holidays
from statsmodels.tsa.stattools import grangercausalitytests

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load_elec_dah_data(self):
        try:
            data = pd.read_csv(self.file_path, 
                            parse_dates=True) # use as datetime format
            return data
        except pd.errors.ParserError as e:
            print(f"Error parsing file: {e}")
            return None

    def load_elec_dah_data_index(self, in_col="date"):
        try:        
            data = pd.read_csv(self.file_path,
                            index_col = in_col, # set date day as index column
                            parse_dates=True)
            return data
        except pd.errors.ParserError as e:
            print(f"Error parsing file: {e}")
            return None


class Preprocessor:
    def __init__(self):
        pass
        
    def find_null_row(df, col_name=None):
        if col_name is not None:
            print(df[df[col_name].isnull()])
        else:
            print("Column name is not provided")
    
    def drop_columns(data, columns):
        """
        Drop columns that are not needed for analysis.
        
        Args:
        data (pd.DataFrame): The DataFrame from which columns will be dropped.
        columns (list): List of column names to be dropped.

        Returns:
        pd.DataFrame: A new DataFrame with specified columns dropped.
        """
        # Drop specified columns and return the new DataFrame
        return data.drop(columns, axis=1, inplace=False)
    
    # Define a filter function
    def filter_hour_02_to_03(df):
            return df[df['hour'] == '02:00 - 03:00']
    
    def fill_null_values(df, col_name, method=None, condition=None):
        # Apply the condition as a filter function if provided
        if condition is not None:
            subset = condition(df)
        else:
            subset = df

        # Fill null values based on the specified method
        if method == "median":
            fill_value = subset[col_name].median()
            df[col_name].fillna(fill_value, inplace=True)
        elif method == "mean":
            fill_value = subset[col_name].mean()
            df[col_name].fillna(fill_value, inplace=True)
        else:
            print("No filling method specified. Please choose 'mean' or 'median'.")

    # Renaming columns to include the unit price
    def rename_columns(df, col_name_scr, col_name_tar):
        if len(col_name_scr) == len(col_name_tar):
            rename_dict = dict(zip(col_name_scr, col_name_tar))
            df.rename(columns=rename_dict, inplace=True)
        else:
            print("The lists of source and target column names must have the same length.")

    # Renaming column
    def rename_col(df, col_name_scr, col_name_tar):
            df.rename(columns={col_name_scr: col_name_tar}, inplace=True)

    def convert_datetime(df):
        # Combine date and hour, then convert to datetime
        # Assuming the format is consistent and the time is always the start time of the hour
        df['Datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['hour'].str[:5])

        return df['Datetime']
    
    # Extract hour for separate analysis if needed
    def extract_hr(df):
        df['Hr'] = df['Datetime'].dt.hour

        return df['Hr']
    
    def remove_outliers(df):
        # Remove outliers using a method like Z-score
        z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
        filtered_entries = (z_scores < 15).all(axis=1)
        return df[filtered_entries]

class FeatureEngineering:
    def __init__(self, df):
        self.df=df
    
    def extract_datetime(self):
        self.df['hour'] = self.df['Datetime'].dt.hour
        self.df['day'] = self.df['Datetime'].dt.day
        self.df['day_of_week'] = self.df['Datetime'].dt.dayofweek  # Monday=0, Sunday=6
        self.df['month'] = self.df['Datetime'].dt.month
        self.df['year'] = self.df['Datetime'].dt.year
        return self

    def add_is_weekend(self):
        self.df['is_weekend'] = self.df['Datetime'].dt.dayofweek >= 5  # 5 for Saturday and 6 for Sunday
        return self

    def add_season(self):
        def get_season(month):
            if month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            elif month in [9, 10, 11]:
                return 'Autumn'
            else:
                return 'Winter'

        self.df['season'] = self.df['Datetime'].dt.month.apply(get_season)
        return self

    def add_public_holiday(self, country='France'):
        country_holidays = holidays.CountryHoliday(country)
        self.df['is_holiday'] = self.df['Datetime'].dt.date.apply(lambda x: x in country_holidays)
        return self
    
    def add_lag_features(self, lags):
        for lag in lags:
            self.df[f'price_lag_{lag}h'] = self.df['hr_elec_price'].shift(lag)
        return self 
       
    def add_rolling_features(self, windows):
        for window in windows:
            self.df[f'rolling_mean_{window}h'] = self.df['hr_elec_price'].rolling(window=window).mean()
            self.df[f'rolling_std_{window}h'] = self.df['hr_elec_price'].rolling(window=window).std()
        return self

    def add_price_change(self):
        self.df['price_change_1h'] = self.df['hr_elec_price'].diff()
        self.df['price_change_24h'] = self.df['hr_elec_price'].diff(24)
        return self

    def get_features(self):
        return self.df


class DataPreparation():
    def __init__(self, df):
        self.df=df

    def prepare_data_lstm(self, target_col, n_input_steps, n_features):

        # One-hot encode 'season' column
        self.df = pd.get_dummies(self.df, columns=['season'], drop_first=True)

        # Convert boolean columns to integer type (True to 1, False to 0)
        self.df['is_weekend'] = self.df['is_weekend'].astype(int)
        self.df['is_holiday'] = self.df['is_holiday'].astype(int)

        datetime_col = self.df['Datetime']
        
        # Exclude the 'Datetime' column from scaling
        features_to_scale = self.df.drop(columns=['Datetime'])
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(features_to_scale)

        # Create sequences
        X, y, dates = [], [], []
        for i in range(n_input_steps, len(scaled_data)):
            X.append(scaled_data[i-n_input_steps:i, :n_features])
            y.append(scaled_data[i, features_to_scale.columns.get_loc(target_col)])
            dates.append(datetime_col.iloc[i])
        
        X, y, dates = np.array(X), np.array(y), np.array(dates)
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        # Align the dates with the test set
        _, test_dates = train_test_split(dates, test_size=0.2, shuffle=False)
        
        return X_train, X_test, y_train, y_test, scaler, test_dates


    def prepare_data_rf_xgb(self, target_col):

        # One-hot encode 'season' column
        self.df = pd.get_dummies(self.df, columns=['season'], drop_first=True)

        # Convert boolean columns to integer type (True to 1, False to 0)
        self.df['is_weekend'] = self.df['is_weekend'].astype(int)
        self.df['is_holiday'] = self.df['is_holiday'].astype(int)
        
        datetime_col = self.df['Datetime']
        
        # Drop the 'Datetime' column from the features
        features_df = self.df.drop(columns=['Datetime'])
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(features_df)

        # Features and target
        X = scaled_data[:, features_df.columns != target_col]
        y = scaled_data[:, features_df.columns.get_loc(target_col)]
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        # Similarly, split the datetime column to align with the test set
        _, test_dates = train_test_split(datetime_col, test_size=0.2, shuffle=False)
        
        return X_train, X_test, y_train, y_test, scaler, test_dates


class LSTMModel:
    def __init__(self, n_input_steps, n_features, n_neurons=50, n_outputs=1, model_path='./data/lstm_model.keras'):
        self.n_input_steps = n_input_steps
        self.n_features = n_features
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        self.model_path = model_path
        self.model = self.build_model()
    
    def build_model(self):
        model = Sequential([
            LSTM(units=self.n_neurons, return_sequences=True, input_shape=(self.n_input_steps, self.n_features), dropout=0.05),
            LSTM(self.n_neurons, return_sequences=True, dropout=0.05),
            LSTM(units=self.n_neurons, dropout=0.1),
            Dense(self.n_outputs)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train(self, X_train, y_train, epochs=100, batch_size=64):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        # Save the model after training
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def load_trained_model(self):
        self.model = load_model(self.model_path)
        print(f"Model loaded from {self.model_path}")
    
    def predict(self, X_test):
        # Ensure the model is loaded before predicting
        if self.model is None:
            self.load_trained_model()
        return self.model.predict(X_test)


class RandomForestModel:
    def __init__(self, n_estimators=100):
        self.model = RandomForestRegressor(n_estimators=n_estimators)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)

class XGBoostModel:
    def __init__(self, n_estimators=100):
        self.model = XGBRegressor(n_estimators=n_estimators)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)


class ModelEvaluator:
    def __init__(self, y_true, y_pred, alg_name=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.alg_name=alg_name
    
    def calculate_mse(self):
        return mean_squared_error(self.y_true, self.y_pred)
    
    def calculate_rmse(self):
        mse = self.calculate_mse()
        return np.sqrt(mse)
    
    def calculate_mae(self):
        return mean_absolute_error(self.y_true, self.y_pred)
    
    def evaluate(self):
        mse = self.calculate_mse()
        rmse = self.calculate_rmse()
        mae = self.calculate_mae()
        print(f"""{self.alg_name}
            MSE: {mse}
            RMSE: {rmse} 
            MAE: {mae}
        """)
    
    def get_metrics(self):
        mse = self.calculate_mse()
        rmse = self.calculate_rmse()
        mae = self.calculate_mae()
        return {'MSE': mse, 'RMSE': rmse, 'MAE': mae}


class GrangerCausality:
    def __init__(self, df):
        self.df = df
    
    def granger_causality_analysis(self, col_name, max_lags=24):

        numeric_df = self.df.select_dtypes(include=[np.number])

        for col in numeric_df.columns:
            if col != col_name:
                print(f"Granger Causality for {col}:")
                gc_test = grangercausalitytests(numeric_df[[col_name, col]], max_lags)

                # Manually print summary if needed
                for lag, test_results in gc_test.items():
                    f_test = test_results[0]['ssr_ftest']
                    print(f"Lag: {lag}, F-Statistic: {f_test[0]}, P-Value: {f_test[1]}")
                print("\n")


class Visualization:
    def __init__(self, df):
        self.df = df
  
    def plot_elec_dah_price_countries(self, countries):

        # Determine the layout of the subplots
        n = len(countries)
        rows = int(n ** 0.5)
        cols = n // rows + (n % rows > 0)

        fig, axes = plt.subplots(rows, cols, figsize=(15, 10), sharex=True)  # Create a grid of subplots
        axes = axes.flatten()  # Flatten the axes array for easy iteration

        for i, country in enumerate(countries):
            ax = axes[i]
            # Group by date, calculate the mean, and plot on the ith subplot
            self.df.groupby('date')[country].mean().plot(ax=ax, title=country)
            ax.set_title(country)  # Set the title to the country name
            ax.set_xlabel('Date')  # Set the x-axis label
            ax.set_ylabel('Average Price')  # Set the y-axis label

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()  # Adjust the layout so labels don't overlap
        plt.show()


    def plot_correlation_matrix(self, countries):

        correlation_matrix = self.df[countries].corr()

        # Plotting the correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
        plt.title('Correlation Matrix of Electricity Prices among Countries')
        plt.show()

    @staticmethod
    def plot_boxplot(df, col_name=None):
        sns.boxplot(df[col_name]) 
        plt.show()

    @staticmethod
    def plot_elec_dah_price(df, col_name):
        color_pal = sns.color_palette()

        #  Time series plot for hourly electricity prices
        plt.figure(figsize=(10, 6))
        # sns.lineplot(data=df, x=df.index, y=col_name, color=color_pal[0])
        sns.scatterplot(data=df, x=df.index, y=col_name, color=color_pal[1], marker='.', s=100)  

        plt.title(f'Line Plot of {col_name}')
        plt.xlabel('Datetime')  
        plt.ylabel('Electricity Price (€/MWh)')
        plt.show()


class Plotter:
    def __init__(self, df):
        self.df=df

    def plot_timeseries(self):
        plt.figure(figsize=(14, 6))
        plt.plot(self.df['Datetime'], self.df['hr_elec_price'], label='Hourly Electricity Price', alpha=0.5)
        plt.plot(self.df['Datetime'], self.df['rolling_mean_24h'], label='24-Hour Rolling Mean', color='red', linewidth=2)
        plt.title('Hourly Electricity Price and 24-Hour Rolling Mean')
        plt.xlabel('Datetime')
        plt.ylabel('Electricity Price (€/MWh)')
        plt.legend()
        plt.show()

    def plot_scatterplot(self):
        plt.figure(figsize=(10, 5))
        plt.scatter(self.df['price_lag_1h'], self.df['hr_elec_price'], alpha=0.5)
        plt.title('Current Price vs. 1-Hour Lagged Price')
        plt.xlabel('1-Hour Lagged Price (€/MWh)')
        plt.ylabel('Current Hourly Electricity Price (€/MWh)')
        plt.show()

    def plot_boxplot_category(self):
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='day_of_week', y='hr_elec_price', data=self.df)
        plt.title('Electricity Price Distribution by Day of the Week')
        plt.xlabel('Day of the Week (0=Monday, 6=Sunday)')
        plt.ylabel('Electricity Price (€/MWh)')
        plt.show()

    def plot_correlation_heatmap(self):
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.df[['hr_elec_price', 'price_lag_1h', 'price_lag_24h', 'rolling_mean_24h', 'rolling_std_24h', 'price_change_1h', 'price_change_24h']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap of Electricity Prices and Derived Features')
        plt.show()

    @staticmethod
    def plot_predict_actual(test_dates, y_test, y_pred, alg_name=None):

        plt.figure(figsize=(15, 7))  # Set the figure size for better visibility

        # Plotting the actual values
        plt.plot(test_dates, y_test, label='Actual Prices', color='blue', marker='.', linestyle='-')

        # Plotting the predicted values
        plt.plot(test_dates, y_pred, label='Predicted Prices', color='red', marker='.', linestyle='-')

        plt.title(f'Actual vs. Predicted Electricity Prices for {alg_name}')  
        plt.xlabel('Date')  # Setting the x-axis label
        plt.ylabel('Electricity Price (€/MWh)')  # Setting the y-axis label
        plt.legend()  # Adding a legend to distinguish between actual and predicted values

        plt.tight_layout()  # Adjust layout to not overlap plot elements
        plt.show()  # Display the plot
