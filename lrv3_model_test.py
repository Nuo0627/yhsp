import pickle
import pandas as pd
import numpy as np

def create_fourier_series_features(prefix, time, period, num_terms):
    terms = np.arange(1, num_terms + 1)
    angular_frequencies = 2 * np.pi * terms / period

    features = pd.DataFrame()
    for i in range(num_terms):
        features[prefix+f'fourier_{i + 1}'] = np.cos(angular_frequencies[i] * time)

    return features

def get_latest_data(sku_id, store_id):
    # Load the latest data from a CSV file, database, etc.
    data = pd.read_csv("latest_data.csv")

    # Convert the 'date' column to datetime
    data['date'] = pd.to_datetime(data['date'])

    # Filter the data for the given SKU and store
    df = data[(data['sku_id'] == sku_id) & (data['store_id'] == store_id)]

    # Feature: Day of Week (0 = Monday, 6 = Sunday)
    df['day_of_week'] = df['date'].dt.dayofweek

    # Feature: Month
    df['month'] = df['date'].dt.month

    # Feature: Quarter
    df['quarter'] = df['date'].dt.quarter

    # Lagged features (previous day's values)
    df['lagged_sales'] = df.groupby(['store_id', 'sku_id'])['sales_amount'].transform(lambda x: x.shift(1))
    df['lagged_in_promotion'] = df.groupby(['store_id', 'sku_id'])['in_promotion'].transform(lambda x: x.shift(1))

    # Rolling mean and standard deviation
    df['rolling_mean_sales'] = df.groupby(['store_id', 'sku_id'])['sales_amount'].transform(lambda x: x.rolling(window=3).mean())
    df['rolling_std_sales'] = df.groupby(['store_id', 'sku_id'])['sales_amount'].transform(lambda x: x.rolling(window=3).std())

    # Average sales, in_promotion, and pb_order at store level
    df['store_avg_sales'] = df.groupby(['store_id'])['sales_amount'].transform('mean')

    # Average sales, in_promotion, and pb_order at SKU level
    df['sku_avg_sales'] = df.groupby(['sku_id'])['sales_amount'].transform('mean')

    # Interaction features
    df['interaction_sales'] = df.apply(lambda x: x['sales_amount'] * x['in_promotion'], axis=1)

    # Drop the 'pb_order' column if it was not used as a feature during training
    df = df.drop(columns=['pb_order'])

    # Feature: Fourier series features for weekly seasonality (7-day cycle)
    df['time'] = df.groupby(['store_id', 'sku_id']).cumcount() + 1
    weekly_fourier_features = create_fourier_series_features("week_", df['time'], period=7, num_terms=5)

    # Feature: Fourier series features for monthly seasonality (30-day cycle)
    monthly_fourier_features = create_fourier_series_features("month_", df['time'], period=30, num_terms=5)

    # Combine all features
    df = pd.concat([df, weekly_fourier_features, monthly_fourier_features], axis=1)

    df = df.fillna(0)

    return df


# Load the dictionary of models from disk
filename = 'models_dict.sav'
models_dict = pickle.load(open(filename, 'rb'))

# Input SKU ID and store ID
sku_id = int(input("Enter SKU ID: "))
store_id = int(input("Enter store ID: "))

# Get the model for the input SKU and store
model = models_dict.get((sku_id, store_id))

if model is None:
    print(f"No model found for SKU {sku_id}, Store {store_id}.")
else:
    # Get the latest data for the input SKU and store
    latest_data = get_latest_data(sku_id, store_id)

    # Drop 'date' and 'sales_amount' columns if they exist
    latest_data = latest_data.drop(columns=['date', 'sales_amount'], errors='ignore')

    #print("latest data",latest_data)

    # Make a prediction
    prediction = model.predict(latest_data)
    prediction = np.maximum(prediction, 0)

    print(f"The predicted sales amount for the next day is: {prediction[0]}")
