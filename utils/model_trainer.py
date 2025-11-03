# utils/model_trainer.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from datetime import datetime, timedelta
import numpy as np
from utils import data_manager # Import data_manager for DB interaction

# Directory to save trained models
MODEL_DIR = 'models'
# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

def train_new_model(product_db_id=None):
    """
    Trains a new sales forecasting model using historical data.
    If `product_db_id` is provided, a specific model for that product is trained.
    Otherwise, a global model is trained using aggregated sales data.

    :param product_db_id: Database ID of the product to train for (optional).
    :return: Tuple of (model_db_id, metrics_dict)
    :raises ValueError: If insufficient data for training.
    """
    print(f"Starting model training for product_db_id: {product_db_id if product_db_id else 'Global'}")

    # 1. Load Data
    if product_db_id:
        sales_data_raw = data_manager.get_sales_data_for_product(product_db_id)
        product_info = data_manager.get_product_by_id(product_db_id)
        model_name_prefix = f"Prod_{product_info['product_id']}" if product_info else "Unknown_Product"
    else:
        # For a global model, aggregate sales data across all products
        # This is a simplified aggregation, a real system might use more sophisticated methods
        all_sales = data_manager.query_db("SELECT sale_date, SUM(quantity_sold) as total_quantity FROM sales_data GROUP BY sale_date ORDER BY sale_date ASC")
        sales_data_raw = all_sales
        model_name_prefix = "Global"

    if not sales_data_raw:
        raise ValueError("Not enough data to train a model. Please upload more sales data.")

    df = pd.DataFrame(sales_data_raw)
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    df = df.set_index('sale_date').sort_index()

    # Rename target column for consistency if it's 'total_quantity' for global
    if product_db_id is None:
        df.rename(columns={'total_quantity': 'quantity_sold'}, inplace=True)

    # 2. Feature Engineering (Simplified Time-Series Features)
    # These features are extracted from the date to help the model learn patterns.
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    df['day_of_year'] = df.index.dayofyear
    # For more advanced time series, consider:
    # - Lag features (e.g., sales from previous month)
    # - Rolling window statistics (e.g., 3-month average sales)
    # - One-hot encoding for month/day_of_week if non-linear effects are expected
    # - External factors (holidays, promotions, weather, etc.)

    # Target variable: quantity_sold
    X = df[['year', 'month', 'day_of_week', 'day_of_year']]
    y = df['quantity_sold']

    if len(df) < 30: # Require a minimum number of data points for meaningful training
        raise ValueError(f"Insufficient data points for training for {model_name_prefix} (need at least 30, found {len(df)}).")
    
    # 3. Split Data (Time-based split for time series)
    # Ensure test set is recent data to evaluate forecasting ability
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    if X_train.empty or X_test.empty:
        raise ValueError("Data split resulted in empty training or testing sets. Adjust data or split logic.")

    # 4. Train Model
    # RandomForestRegressor is a good general-purpose model for tabular data
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # Use all CPU cores
    model.fit(X_train, y_train)

    # 5. Evaluate Model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r_squared = r2_score(y_test, y_pred)

    print(f"Model for {model_name_prefix} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r_squared:.3f}")

    # 6. Save Model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_name_prefix}_RF_{timestamp}.joblib"
    file_path = os.path.join(MODEL_DIR, model_filename)
    joblib.dump(model, file_path)

    # 7. Store Model Metadata in DB
    model_data = {
        'model_name': f"{model_name_prefix}_RF_v{mae:.0f}", # Simple versioning based on MAE
        'training_date': datetime.now(),
        'mae': mae,
        'rmse': rmse,
        'r_squared': r_squared,
        'status': 'inactive', # Initially inactive, admin needs to activate
        'file_path': file_path,
        'product_id': product_db_id # NULL for global models
    }
    model_db_id = data_manager.add_model(model_data)
    print(f"Model metadata saved to DB with ID: {model_db_id}")

    return model_db_id, {'mae': round(mae, 2), 'rmse': round(rmse, 2), 'r_squared': round(r_squared, 3)}

def generate_forecast(model_db_id, product_db_id, num_periods=6):
    """
    Generates a sales forecast for a given product using a specified model.
    The forecast is for `num_periods` into the future, starting from the month
    after the last historical sales data point.

    :param model_db_id: The database ID of the model to use for forecasting.
    :param product_db_id: The database ID of the product to forecast for.
    :param num_periods: Number of future months to forecast.
    :return: A list of dictionaries, each containing 'forecast_date' and 'forecasted_quantity'.
    :raises ValueError: If model or product not found, or model file is missing.
    """
    model_meta = data_manager.get_model_by_id(model_db_id)
    if not model_meta:
        raise ValueError(f"Model with ID {model_db_id} not found in database.")

    model_path = model_meta['file_path']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please check model storage.")

    model = joblib.load(model_path)

    # Determine the start date for forecasting
    last_sale = data_manager.query_db(
        "SELECT MAX(sale_date) as last_date FROM sales_data WHERE product_id = %s",
        (product_db_id,),
        one=True
    )

    if last_sale and last_sale['last_date']:
        # Start from the 1st of the month after the last recorded sale
        start_date = (last_sale['last_date'] + timedelta(days=1)).replace(day=1)
    else:
        # If no historical sales, start forecasting from the 1st of the current month
        start_date = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    # Generate future dates (first day of each month)
    forecast_dates = []
    current_forecast_date = start_date
    for _ in range(num_periods):
        forecast_dates.append(current_forecast_date)
        # Advance to the first day of the next month
        if current_forecast_date.month == 12:
            current_forecast_date = current_forecast_date.replace(year=current_forecast_date.year + 1, month=1)
        else:
            current_forecast_date = current_forecast_date.replace(month=current_forecast_date.month + 1)

    # Create future features for prediction
    future_data = pd.DataFrame({
        'sale_date': forecast_dates
    })
    future_data['year'] = future_data['sale_date'].dt.year
    future_data['month'] = future_data['sale_date'].dt.month
    future_data['day_of_week'] = future_data['sale_date'].dt.dayofweek # Consistent with training features
    future_data['day_of_year'] = future_data['sale_date'].dt.dayofyear

    X_future = future_data[['year', 'month', 'day_of_week', 'day_of_year']]

    # Predict
    forecasted_quantities = model.predict(X_future)
    # Ensure non-negative predictions and round to nearest integer
    forecasted_quantities = np.maximum(0, np.round(forecasted_quantities)).astype(int)

    forecast_results = []
    for i in range(num_periods):
        forecast_results.append({
            'forecast_date': forecast_dates[i].date(),
            'forecasted_quantity': int(forecasted_quantities[i]) # Ensure it's a standard Python int
        })

    # Save forecasts to DB
    data_manager.save_forecast(model_db_id, product_db_id, forecast_results)
    return forecast_results
