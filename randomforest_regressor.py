import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Assuming df is your DataFrame with the engineered features
df = pd.read_csv("featured_combined_data.csv")

# List of unique SKU IDs and store IDs
sku_ids = df['sku_id'].unique()
store_ids = df['store_id'].unique()

# Initialize dictionaries to store model results
mse_dict = {}
r2_dict = {}

# Loop through each combination of SKU and store
for sku_id in sku_ids:
    for store_id in store_ids:
        # Filter data for the current SKU and store
        sku_store_data = df[(df['sku_id'] == sku_id) & (df['store_id'] == store_id)].copy()

        # Check if there are enough samples for the SKU-store combination
        if len(sku_store_data) < 2:
            print(f"Skipping SKU {sku_id}, Store {store_id} due to insufficient samples.")
            continue

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            sku_store_data.drop(['date', 'sales_amount'], axis=1),  # Features excluding 'date' and 'sales_amount'
            sku_store_data['sales_amount'],
            test_size=0.2,
            random_state=42
        )

        # Initialize and fit the Random Forest Regressor model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate model performance
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Store results in dictionaries
        mse_dict[(sku_id, store_id)] = mse
        r2_dict[(sku_id, store_id)] = r2

        # Print results for the current SKU and store
        print(f"SKU {sku_id}, Store {store_id}:")
        print(f"Mean Squared Error: {mse}")
        print(f"R-squared: {r2}")
        print("")

# Overall model evaluation (average across SKU-store combinations)
avg_mse = sum(mse_dict.values()) / len(mse_dict)
avg_r2 = sum(r2_dict.values()) / len(r2_dict)

print("Overall Model Evaluation:")
print(f"Average Mean Squared Error: {avg_mse}")
print(f"Average R-squared: {avg_r2}")

