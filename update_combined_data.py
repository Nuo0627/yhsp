import pandas as pd

# Read your data (replace 'your_data.csv' with your actual file name)
df = pd.read_csv('combined_data.csv')

# Update 'sales_amount' based on 'pb_order'
df['sales_amount'] += df['pb_order']

# Drop the 'pb_order' column
df = df.drop('pb_order', axis=1)

# Save the updated data to a new CSV file (replace 'updated_data.csv' with your desired file name)
df.to_csv('updated_combined_data.csv', index=False)
