import pandas as pd
import xgboost as xgb
import joblib

# Load and preprocess your dataset
df = pd.read_csv('retail_store_inventory.csv')
df = df.dropna()
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Feature and target selection
X = df[['Store ID', 'Product ID', 'Category', 'Region', 'Inventory Level', 'Units Sold',
        'Units Ordered', 'Price', 'Discount', 'Competitor Pricing', 'Seasonality', 'Day', 'Month', 'Year']]
y = df['Demand Forecast']

# Encode categorical variables
X = pd.get_dummies(X)

# Train model
model = xgb.XGBRegressor()
model.fit(X, y)

# Save model and feature columns
joblib.dump((model, X.columns), 'model/xgb_model.pkl')
