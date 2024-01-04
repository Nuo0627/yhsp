import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tqdm import tqdm

# Assuming df is your DataFrame with the engineered features
df = pd.read_csv("featured_combined_data.csv")

# List of unique SKU IDs and store IDs
sku_ids = df['sku_id'].unique()
store_ids = df['store_id'].unique()

# Initialize dictionaries to store model results
mse_dict = {}
r2_dict = {}

# Loop through each combination of SKU and store
for sku_id in tqdm(sku_ids, desc="SKU Loop"):
    for store_id in tqdm(store_ids, desc="Store Loop", leave=False):
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

        # Standardize the input features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Initialize the neural network model
        model = Sequential([
            Dense(64, activation='relu', input_dim=X_train.shape[1]),
            Dense(32, activation='relu'),
            Dense(1)  # Output layer with a single neuron for regression
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)

        # Make predictions on the test set
        y_pred = model.predict(X_test_scaled).flatten()

        # Evaluate model performance
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Store results in dictionaries
        mse_dict[(sku_id, store_id)] = mse
        r2_dict[(sku_id, store_id)] = r2

        # Print results for the current SKU and store
        tqdm.write(f"SKU {sku_id}, Store {store_id}:")
        tqdm.write(f"Mean Squared Error: {mse}")
        tqdm.write(f"R-squared: {r2}")
        tqdm.write("")

# Overall model evaluation (average across SKU-store combinations)
avg_mse = sum(mse_dict.values()) / len(mse_dict)
avg_r2 = sum(r2_dict.values()) / len(r2_dict)

print("Overall Model Evaluation:")
print(f"Average Mean Squared Error: {avg_mse}")
print(f"Average R-squared: {avg_r2}")

