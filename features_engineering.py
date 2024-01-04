import pandas as pd

df = pd.read_csv("combined_data.csv")

# Assuming df is your original DataFrame
# Make sure 'date' column is in datetime format
df['date'] = pd.to_datetime(df['date'])

# Feature: Day of Week (0 = Monday, 6 = Sunday)
df['day_of_week'] = df['date'].dt.dayofweek

# Feature: Month
df['month'] = df['date'].dt.month

# Feature: Quarter
df['quarter'] = df['date'].dt.quarter

# Feature: Year
df['year'] = df['date'].dt.year

# Lagged features (previous day's values)
df['lagged_sales'] = df.groupby(['store_id', 'sku_id'])['sales_amount'].shift(1)
df['lagged_in_promotion'] = df.groupby(['store_id', 'sku_id'])['in_promotion'].shift(1)
df['lagged_pb_order'] = df.groupby(['store_id', 'sku_id'])['pb_order'].shift(1)

# Rolling mean and standard deviation
df['rolling_mean_sales'] = df.groupby(['store_id', 'sku_id'])['sales_amount'].transform(lambda x: x.rolling(window=3).mean())
df['rolling_std_sales'] = df.groupby(['store_id', 'sku_id'])['sales_amount'].transform(lambda x: x.rolling(window=3).std())

# Average sales, in_promotion, and pb_order at store level
df['store_avg_sales'] = df.groupby(['store_id'])['sales_amount'].transform('mean')
df['store_avg_in_promotion'] = df.groupby(['store_id'])['in_promotion'].transform('mean')
df['store_avg_pb_order'] = df.groupby(['store_id'])['pb_order'].transform('mean')

# Average sales, in_promotion, and pb_order at SKU level
df['sku_avg_sales'] = df.groupby(['sku_id'])['sales_amount'].transform('mean')
df['sku_avg_in_promotion'] = df.groupby(['sku_id'])['in_promotion'].transform('mean')
df['sku_avg_pb_order'] = df.groupby(['sku_id'])['pb_order'].transform('mean')

# Interaction features
df['interaction_sales'] = df['sales_amount'] * df['in_promotion']
df['interaction_in_promotion'] = df['in_promotion'] * df['pb_order']
df['interaction_pb_order'] = df['pb_order'] * df['sales_amount']

# Drop rows with NaN values (resulting from lagged features)
df = df.dropna()

# Save the DataFrame to a new CSV file
df.to_csv("featured_combined_data.csv", index=False)

