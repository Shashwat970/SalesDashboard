# utils/data_manager.py
import pandas as pd
from database import get_db, execute_db, query_db
from datetime import datetime

def get_all_products():
    """Fetches all products from the database."""
    return query_db("SELECT * FROM products ORDER BY name ASC")

def get_product_by_id(product_id):
    """Fetches a single product by its database ID."""
    return query_db("SELECT * FROM products WHERE id = %s", (product_id,), one=True)

def get_product_by_product_id_string(product_id_str):
    """Fetches a single product by its unique product_id string (e.g., 'P-1001')."""
    return query_db("SELECT * FROM products WHERE product_id = %s", (product_id_str,), one=True)

def add_product(product_data):
    """Adds a new product to the database."""
    return execute_db(
        "INSERT INTO products (product_id, name, category, price, stock, date_added) VALUES (%s, %s, %s, %s, %s, %s)",
        (product_data['product_id'], product_data['name'], product_data['category'],
         product_data['price'], product_data['stock'], product_data['date_added'])
    )

def update_product(product_db_id, product_data):
    """Updates an existing product in the database by its database ID."""
    execute_db(
        "UPDATE products SET name = %s, category = %s, price = %s, stock = %s WHERE id = %s",
        (product_data['name'], product_data['category'], product_data['price'],
         product_data['stock'], product_db_id)
    )

def delete_product(product_db_id):
    """Deletes a product from the database by its database ID."""
    execute_db("DELETE FROM products WHERE id = %s", (product_db_id,))

def get_product_categories():
    """Fetches all unique product categories."""
    categories = query_db("SELECT DISTINCT category FROM products ORDER BY category ASC")
    return [c['category'] for c in categories]

def get_sales_data_for_product(product_db_id):
    """Fetches historical sales data for a given product by its database ID."""
    return query_db(
        "SELECT sale_date, quantity_sold, revenue FROM sales_data WHERE product_id = %s ORDER BY sale_date ASC",
        (product_db_id,)
    )

def upload_sales_data(df, uploaded_by_user_id):
    """
    Uploads sales data from a pandas DataFrame to the database.
    It expects 'product_id', 'sale_date', 'quantity_sold', 'revenue' columns.
    It attempts to match products by their `product_id` string. If a product
    doesn't exist, a new one is created (simplified for demo).
    """
    db, cursor = get_db()
    inserted_rows = 0
    updated_rows = 0
    errors = []

    for index, row in df.iterrows():
        try:
            product_id_str = str(row.get('product_id'))
            if not product_id_str:
                errors.append(f"Row {index}: Missing 'product_id'. Skipping.")
                continue

            product = get_product_by_product_id_string(product_id_str)
            current_product_db_id = None

            if product:
                current_product_db_id = product['id']
            else:
                # Create a new product if not found
                new_product_name = str(row.get('product_name', f"Unknown Product {product_id_str}"))
                new_category = str(row.get('category', 'Uncategorized'))
                new_price = float(row.get('price', 0.00))
                new_stock = int(row.get('stock', 0))
                new_date_added = datetime.now().strftime('%Y-%m-%d')

                new_product_data = {
                    'product_id': product_id_str,
                    'name': new_product_name,
                    'category': new_category,
                    'price': new_price,
                    'stock': new_stock,
                    'date_added': new_date_added
                }
                current_product_db_id = add_product(new_product_data)
                # print(f"Created new product: {new_product_name} with DB ID {current_product_db_id}")

            if current_product_db_id:
                sale_date = pd.to_datetime(row['sale_date']).date()
                quantity_sold = int(row['quantity_sold'])
                revenue = float(row['revenue'])

                cursor.execute(
                    "INSERT INTO sales_data (product_id, sale_date, quantity_sold, revenue) VALUES (%s, %s, %s, %s) "
                    "ON DUPLICATE KEY UPDATE quantity_sold = %s, revenue = %s",
                    (current_product_db_id, sale_date, quantity_sold, revenue, quantity_sold, revenue)
                )
                if cursor.rowcount == 1: # Rowcount is 1 for insert
                    inserted_rows += 1
                elif cursor.rowcount == 2: # Rowcount is 2 for update (matched and changed)
                    updated_rows += 1
        except Exception as e:
            errors.append(f"Row {index} (Product ID: {row.get('product_id', 'N/A')}): Error processing data - {e}")
            db.rollback() # Rollback the current transaction if an error occurs

    db.commit()
    return inserted_rows, updated_rows, errors

def get_all_models():
    """Fetches all trained models metadata, including product name if associated."""
    return query_db("SELECT m.*, p.name AS product_name FROM models m LEFT JOIN products p ON m.product_id = p.id ORDER BY training_date DESC")

def get_model_by_id(model_db_id):
    """Fetches a single model by its database ID."""
    return query_db("SELECT m.*, p.name AS product_name FROM models m LEFT JOIN products p ON m.product_id = p.id WHERE m.id = %s", (model_db_id,), one=True)

def add_model(model_data):
    """Adds new model metadata to the database."""
    return execute_db(
        "INSERT INTO models (model_name, training_date, mae, rmse, r_squared, status, file_path, product_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (model_data['model_name'], model_data['training_date'], model_data['mae'],
         model_data['rmse'], model_data['r_squared'], model_data['status'],
         model_data['file_path'], model_data.get('product_id'))
    )

def update_model_status(model_db_id, status):
    """Updates the status of a model by its database ID."""
    execute_db("UPDATE models SET status = %s WHERE id = %s", (status, model_db_id))

def set_active_model(model_db_id, product_db_id=None):
    """
    Sets a specific model as 'active'.
    If `product_db_id` is provided, all other models for that product become 'inactive'.
    If `product_db_id` is None, all other GLOBAL models (product_id IS NULL) become 'inactive'.
    """
    db, cursor = get_db()
    if product_db_id:
        cursor.execute("UPDATE models SET status = 'inactive' WHERE product_id = %s", (product_db_id,))
    else:
        # For a global model, deactivate all previously active global models
        cursor.execute("UPDATE models SET status = 'inactive' WHERE product_id IS NULL")
    cursor.execute("UPDATE models SET status = 'active' WHERE id = %s", (model_db_id,))
    db.commit()

def delete_model(model_db_id):
    """Deletes model metadata and associated forecasts from the database."""
    # The ON DELETE CASCADE in `forecasts` table handles deleting related forecasts.
    execute_db("DELETE FROM models WHERE id = %s", (model_db_id,))

def get_active_model(product_db_id=None):
    """
    Retrieves the active model for a specific product or the global active model.
    Prioritizes product-specific active model.
    """
    if product_db_id:
        # Try to find an active model specifically for this product
        model = query_db("SELECT * FROM models WHERE product_id = %s AND status = 'active'", (product_db_id,), one=True)
        if model:
            return model
    # If no product_db_id or no product-specific active model, look for a global active model
    return query_db("SELECT * FROM models WHERE product_id IS NULL AND status = 'active'", one=True)

def save_forecast(model_db_id, product_db_id, forecast_data):
    """Saves generated forecast data to the database."""
    db, cursor = get_db()
    for row in forecast_data:
        try:
            cursor.execute(
                "INSERT INTO forecasts (model_id, product_id, forecast_date, forecasted_quantity) VALUES (%s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE forecasted_quantity = %s",
                (model_db_id, product_db_id, row['forecast_date'], row['forecasted_quantity'], row['forecasted_quantity'])
            )
        except Exception as e:
            print(f"Error saving forecast for product {product_db_id}, date {row['forecast_date']}: {e}")
            db.rollback() # Rollback the current transaction for this row
            continue
    db.commit()

def get_forecasts_for_product(product_db_id, model_db_id=None):
    """
    Retrieves forecasts for a specific product.
    If `model_db_id` is provided, fetches forecasts for that model.
    Otherwise, attempts to fetch forecasts from the currently active model for that product/globally.
    """
    if model_db_id:
        return query_db(
            "SELECT forecast_date, forecasted_quantity FROM forecasts WHERE product_id = %s AND model_id = %s ORDER BY forecast_date ASC",
            (product_db_id, model_db_id)
        )
    else:
        active_model = get_active_model(product_db_id)
        if active_model:
            return query_db(
                "SELECT forecast_date, forecasted_quantity FROM forecasts WHERE product_id = %s AND model_id = %s ORDER BY forecast_date ASC",
                (product_db_id, active_model['id'])
            )
    return []

def get_latest_uploads_summary(limit=10):
    """
    Fetches a summary of recent uploads.
    For this demo, we'll return a simplified list based on sales_data entries,
    and associate them with the admin user for 'uploaded_by'.
    In a real system, you'd have an 'uploads' table with more metadata.
    """
    # This is a simplified view for the demo.
    # In a real system, you would have an 'uploads_log' table storing metadata
    # like filename, records, status, uploaded_by_user_id, upload_timestamp.
    # Here, we're mimicking by grouping sales data by product and date.
    raw_uploads = query_db("""
        SELECT
            p.name AS file_name,
            COUNT(sd.id) AS records,
            MAX(sd.sale_date) AS date,
            'Processed' AS status,
            'Admin' AS uploaded_by -- Mocked: in a real app, track uploader
        FROM sales_data sd
        JOIN products p ON sd.product_id = p.id
        GROUP BY p.name, DATE(sd.sale_date)
        ORDER BY MAX(sd.sale_date) DESC
        LIMIT %s
    """, (limit,))

    # Convert to expected format for frontend
    formatted_uploads = []
    for i, upload in enumerate(raw_uploads):
        formatted_uploads.append({
            'id': i + 1, # Dummy ID
            'name': f"{upload['file_name']}_sales_{upload['date'].strftime('%Y%m')}.csv",
            'records': upload['records'],
            'by': upload['uploaded_by'],
            'date': upload['date'].strftime('%Y-%m-%d %H:%M'), # Mock time for consistency
            'status': upload['status']
        })
    return formatted_uploads
