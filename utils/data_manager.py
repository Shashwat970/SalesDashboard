# utils/data_manager.py
import pandas as pd
from database import get_db, execute_db, query_db
from datetime import datetime


# -------------------- PRODUCTS --------------------

def get_all_products():
    return query_db("""
        SELECT id, product_id, name AS product_name, category,
               price, stock, date_added AS dateAdded
        FROM products
        ORDER BY date_added DESC
    """)


def get_product_by_id(product_id):
    return query_db("SELECT * FROM products WHERE id = ?", (product_id,), one=True)


def get_product_by_product_id_string(product_id_str):
    return query_db("SELECT * FROM products WHERE product_id = ?", (product_id_str,), one=True)


def add_product(product_data):
    return execute_db(
        """INSERT INTO products 
        (product_id, name, category, price, stock, date_added)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (
            product_data['product_id'],
            product_data['name'],
            product_data['category'],
            product_data['price'],
            product_data['stock'],
            product_data['date_added']
        )
    )


def update_product(product_db_id, product_data):
    execute_db(
        """UPDATE products 
        SET name = ?, category = ?, price = ?, stock = ?
        WHERE id = ?""",
        (
            product_data['name'],
            product_data['category'],
            product_data['price'],
            product_data['stock'],
            product_db_id
        )
    )


def delete_product(product_db_id):
    execute_db("DELETE FROM products WHERE id = ?", (product_db_id,))


def get_product_categories():
    categories = query_db("SELECT DISTINCT category FROM products ORDER BY category ASC")
    return [c['category'] for c in categories]


# -------------------- SALES DATA --------------------

def get_sales_data_for_product(product_db_id):
    return query_db(
        """SELECT sale_date, quantity_sold, revenue
           FROM sales_data
           WHERE product_id = ?
           ORDER BY sale_date ASC""",
        (product_db_id,)
    )


def upload_sales_data(df, uploaded_by_user_id):
    db = get_db()
    cursor = db.cursor()

    inserted_rows = 0
    errors = []

    for index, row in df.iterrows():
        try:
            # ---- Validate product_id ----
            product_id_raw = row.get('product_id')

            if pd.isna(product_id_raw) or str(product_id_raw).strip() == "":
                errors.append(f"Row {index}: Missing product_id")
                continue

            product_id_str = str(product_id_raw).strip()

            product = get_product_by_product_id_string(product_id_str)

            # ---- Create product if not exists ----
            if product:
                product_db_id = product['id']
            else:
                product_db_id = add_product({
                    'product_id': product_id_str,
                    'name': str(row.get('product_name', f"Product {product_id_str}")),
                    'category': str(row.get('category', 'Uncategorized')),
                    'price': float(row.get('price', 0.0)),
                    'stock': int(row.get('stock', 0)),
                    'date_added': datetime.now().strftime('%Y-%m-%d')
                })

            # ---- Insert sales data ----
            sale_date = pd.to_datetime(row['sale_date']).strftime('%Y-%m-%d')
            quantity_sold = int(row['quantity_sold'])
            revenue = float(row['revenue'])

            cursor.execute("""
                INSERT OR REPLACE INTO sales_data
                (product_id, sale_date, quantity_sold, revenue)
                VALUES (?, ?, ?, ?)
            """, (product_db_id, sale_date, quantity_sold, revenue))

            inserted_rows += 1

        except Exception as e:
            errors.append(f"Row {index}: {str(e)}")
            continue  # DO NOT rollback entire DB

    db.commit()
    return inserted_rows, 0, errors


# -------------------- MODELS --------------------

def get_all_models():
    return query_db("""
        SELECT m.*, p.name AS product_name
        FROM models m
        LEFT JOIN products p ON m.product_id = p.id
        ORDER BY training_date DESC
    """)


def get_model_by_id(model_db_id):
    return query_db("""
        SELECT m.*, p.name AS product_name
        FROM models m
        LEFT JOIN products p ON m.product_id = p.id
        WHERE m.id = ?
    """, (model_db_id,), one=True)


def add_model(model_data):
    return execute_db("""
        INSERT INTO models
        (model_name, training_date, mae, rmse, r_squared, status, file_path, product_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        model_data['model_name'],
        model_data['training_date'],
        model_data['mae'],
        model_data['rmse'],
        model_data['r_squared'],
        model_data['status'],
        model_data['file_path'],
        model_data.get('product_id')
    ))


def update_model_status(model_db_id, status):
    execute_db("UPDATE models SET status = ? WHERE id = ?", (status, model_db_id))


def set_active_model(model_db_id, product_db_id=None):
    db = get_db()
    cursor = db.cursor()

    if product_db_id:
        cursor.execute("UPDATE models SET status = 'inactive' WHERE product_id = ?", (product_db_id,))
    else:
        cursor.execute("UPDATE models SET status = 'inactive' WHERE product_id IS NULL")

    cursor.execute("UPDATE models SET status = 'active' WHERE id = ?", (model_db_id,))
    db.commit()


def delete_model(model_db_id):
    execute_db("DELETE FROM models WHERE id = ?", (model_db_id,))


def get_active_model(product_db_id=None):
    if product_db_id:
        model = query_db(
            "SELECT * FROM models WHERE product_id = ? AND status = 'active'",
            (product_db_id,), one=True
        )
        if model:
            return model

    return query_db(
        "SELECT * FROM models WHERE product_id IS NULL AND status = 'active'",
        one=True
    )


# -------------------- FORECAST --------------------

def save_forecast(model_db_id, product_db_id, forecast_data):
    db = get_db()
    cursor = db.cursor()

    for row in forecast_data:
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO forecasts
                (model_id, product_id, forecast_date, forecasted_quantity)
                VALUES (?, ?, ?, ?)
            """, (
                model_db_id,
                product_db_id,
                row['forecast_date'],
                row['forecasted_quantity']
            ))
        except Exception as e:
            print(f"Forecast save error: {e}")
            continue

    db.commit()


def get_forecasts_for_product(product_db_id, model_db_id=None):
    if model_db_id:
        return query_db("""
            SELECT forecast_date, forecasted_quantity
            FROM forecasts
            WHERE product_id = ? AND model_id = ?
            ORDER BY forecast_date ASC
        """, (product_db_id, model_db_id))

    active_model = get_active_model(product_db_id)

    if active_model:
        return query_db("""
            SELECT forecast_date, forecasted_quantity
            FROM forecasts
            WHERE product_id = ? AND model_id = ?
            ORDER BY forecast_date ASC
        """, (product_db_id, active_model['id']))

    return []


# -------------------- UPLOAD SUMMARY --------------------

def get_latest_uploads_summary(limit=10):
    raw_uploads = query_db("""
        SELECT
            p.name AS file_name,
            COUNT(sd.id) AS records,
            MAX(sd.sale_date) AS date,
            'Processed' AS status,
            'Admin' AS uploaded_by
        FROM sales_data sd
        JOIN products p ON sd.product_id = p.id
        GROUP BY p.name, DATE(sd.sale_date)
        ORDER BY MAX(sd.sale_date) DESC
        LIMIT ?
    """, (limit,))

    formatted_uploads = []

    for i, upload in enumerate(raw_uploads):
        date_val = upload['date']

        if isinstance(date_val, str):
            date_val = pd.to_datetime(date_val)

        formatted_uploads.append({
            'id': i + 1,
            'name': f"{upload['file_name']}_sales_{date_val.strftime('%Y%m')}.csv",
            'records': upload['records'],
            'by': upload['uploaded_by'],
            'date': date_val.strftime('%Y-%m-%d %H:%M'),
            'status': upload['status']
        })

    return formatted_uploads
