import pandas as pd
import matplotlib.pyplot as plt

# Load your sales data for a specific SKU
sku_sales_data = pd.read_csv('368802641_sales_data.csv')  # Replace with your actual file name

# Assuming you have columns: date, store_id, sku_id, sales_amount
# Convert 'date' column to datetime format
sku_sales_data['date'] = pd.to_datetime(sku_sales_data['date'])

# Create a unique color for each store
unique_stores = sku_sales_data['store_id'].unique()
store_color_mapping = {store_id: plt.cm.viridis(i / len(unique_stores)) for i, store_id in enumerate(unique_stores)}

# Plot the sales amount for each store with different colors
plt.figure(figsize=(64, 32))  # Larger figure size
for store_id, group in sku_sales_data.groupby('store_id'):
    plt.plot(group['date'], group['sales_amount'], label=f'Store {store_id}', marker='o', linestyle='-', color=store_color_mapping[store_id], markersize=2, linewidth=0.5)

plt.title('Sales Amount Over Time for SKU 368802641')
plt.xlabel('Date')
plt.ylabel('Sales Amount')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Move the legend to the side
plt.grid(True)

# Save the plot as a PNG file
plt.savefig('sales_plot_multi_store_large.png')

# Print a message indicating where the plot is saved
print("Plot saved as 'sales_plot_multi_store_large.png'")

