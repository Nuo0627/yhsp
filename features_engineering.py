import pandas as pd
import numpy as np
from tqdm import tqdm

def create_fourier_series_features(prefix, time, period, num_terms):
    """
    Create Fourier series features for time series data.

    Parameters:
    - time: The time values.
    - period: The period of the cycle (e.g., 7 for weekly, 365 for yearly).
    - num_terms: The number of Fourier terms to create.

    Returns:
    - A DataFrame with Fourier series features.
    """
    terms = np.arange(1, num_terms + 1)
    angular_frequencies = 2 * np.pi * terms / period

    features = pd.DataFrame()
    for i in range(num_terms):
        features[prefix+f'fourier_{i + 1}'] = np.cos(angular_frequencies[i] * time)

    return features

df = pd.read_csv("updated_combined_data.csv")

# Assuming df is your original DataFrame
# Make sure 'date' column is in datetime format
df['date'] = pd.to_datetime(df['date'])

# Enable progress_apply for Pandas Series
tqdm.pandas()

# Wrap the entire block of code with tqdm context manager
with tqdm(total=11, desc="Processing Data") as pbar:
    # Feature: Day of Week (0 = Monday, 6 = Sunday)
    df['day_of_week'] = df['date'].dt.dayofweek
    pbar.update(1)

    # Feature: Month
    df['month'] = df['date'].dt.month
    pbar.update(1)

    # Feature: Quarter
    df['quarter'] = df['date'].dt.quarter
    pbar.update(1)

    # Lagged features (previous day's values)
    df['lagged_sales'] = df.groupby(['store_id', 'sku_id'])['sales_amount'].transform(lambda x: x.shift(1))
    df['lagged_in_promotion'] = df.groupby(['store_id', 'sku_id'])['in_promotion'].transform(lambda x: x.shift(1))
    pbar.update(2)

    # Rolling mean and standard deviation
    df['rolling_mean_sales'] = df.groupby(['store_id', 'sku_id'])['sales_amount'].transform(lambda x: x.rolling(window=3).mean())
    df['rolling_std_sales'] = df.groupby(['store_id', 'sku_id'])['sales_amount'].transform(lambda x: x.rolling(window=3).std())
    pbar.update(2)

    # Average sales, in_promotion, and pb_order at store level
    df['store_avg_sales'] = df.groupby(['store_id'])['sales_amount'].transform('mean').progress_apply(lambda x: x)
    pbar.update(1)

    # Average sales, in_promotion, and pb_order at SKU level
    df['sku_avg_sales'] = df.groupby(['sku_id'])['sales_amount'].transform('mean').progress_apply(lambda x: x)
    pbar.update(1)

    # Interaction features
    df['interaction_sales'] = df.progress_apply(lambda x: x['sales_amount'] * x['in_promotion'], axis=1)
    pbar.update(1)

    # Feature: Fourier series features for weekly seasonality (7-day cycle)
    df['time'] = df.groupby(['store_id', 'sku_id']).cumcount() + 1
    weekly_fourier_features = create_fourier_series_features("week_", df['time'], period=7, num_terms=5)
    pbar.update(1)

    # Feature: Fourier series features for monthly seasonality (30-day cycle)
    monthly_fourier_features = create_fourier_series_features("month_", df['time'], period=30, num_terms=5)
    pbar.update(1)

# Combine all features
df = pd.concat([df, weekly_fourier_features, monthly_fourier_features], axis=1)

# Create interaction features among SKUs
# sku_list = [368802559, 368802586, 368802608]  # Add the SKUs you are interested in
# for sku in sku_list:
#     df[f'interaction_sales_sku_{sku}'] = df['sales_amount'] * (df['sku_id'] == sku)

# Drop rows with NaN values (resulting from lagged features)
df = df.fillna(0)

# Save the DataFrame to a new CSV file
df.to_csv("featured_combined_data.csv", index=False)
