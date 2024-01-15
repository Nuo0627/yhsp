import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import tqdm
import pickle

# Assuming df is your DataFrame with the engineered features
df = pd.read_csv("featured_combined_data.csv")

# List of unique SKU IDs and store IDs
sku_ids = df['sku_id'].unique()
store_ids = df['store_id'].unique()

# Initialize dictionaries to store model results and models
mse_dict = {}
r2_dict = {}
models_dict = {}

# Create a progress bar for the outer loop
outer_pbar = tqdm.tqdm(total=len(sku_ids), desc="SKU Progress")

# Loop through each combination of SKU and store
for sku_id in tqdm.tqdm(sku_ids, desc="SKU Progress"):
    for store_id in tqdm.tqdm(store_ids, desc="Store Progress", leave=False):
        # Filter data for the current SKU and store
        sku_store_data = df[(df['sku_id'] == sku_id) & (df['store_id'] == store_id)].copy()

        # Check if there are enough samples for the SKU-store combination
        if len(sku_store_data) < 2:
            tqdm.tqdm.write(f"Skipping SKU {sku_id}, Store {store_id} due to insufficient samples.")
            continue

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            sku_store_data.drop(['date', 'sales_amount'], axis=1),  # Features excluding 'date' and 'sales_amount'
            sku_store_data['sales_amount'],
            test_size=0.2,
            random_state=42
        )

        # Check if there are enough samples for cross-validation
        if len(X_train) < 5:  # Adjust the threshold as needed
            tqdm.tqdm.write(f"Skipping cross-validation for SKU {sku_id}, Store {store_id} due to insufficient training samples.")
            continue

        # Create a pipeline with PolynomialFeatures and Ridge regression
        degree = 2  # Degree of polynomial features
        alpha = 0.1  # Regularization strength

        model = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree),
            Ridge(alpha=alpha)
        )

        # Perform cross-validation to evaluate the model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_mse = -cv_scores.mean()  # Taking the negative because cross_val_score returns negative MSE

        # Fit the model on the training data
        model.fit(X_train, y_train)

        # Store the model in the dictionary
        models_dict[(sku_id, store_id)] = model

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate model performance
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Store results in dictionaries
        mse_dict[(sku_id, store_id)] = mse
        r2_dict[(sku_id, store_id)] = r2

        # Print results for the current SKU and store
        tqdm.tqdm.write(f"SKU {sku_id}, Store {store_id}:")
        tqdm.tqdm.write(f"Mean Squared Error: {mse}")
        tqdm.tqdm.write(f"R-squared: {r2}")
        tqdm.tqdm.write(f"Cross-validated Mean Squared Error: {cv_mse}")
        tqdm.tqdm.write("")

# Close the progress bar for the outer loop
outer_pbar.close()

# Overall model evaluation (average across SKU-store combinations)
avg_mse = sum(mse_dict.values()) / len(mse_dict)
avg_r2 = sum(r2_dict.values()) / len(r2_dict)

print("Overall Model Evaluation:")
print(f"Average Mean Squared Error: {avg_mse}")
print(f"Average R-squared: {avg_r2}")

# Save the dictionary to disk
filename = 'full_models_dict.sav'
pickle.dump(models_dict, open(filename, 'wb'))
