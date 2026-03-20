import sqlite3
from flask import current_app, g
from werkzeug.security import generate_password_hash

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(current_app.config['DATABASE'])
        g.db.row_factory = sqlite3.Row
    return g.db

def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db(app):
    with app.app_context():
        db = get_db()
        cursor = db.cursor()

        # SQLite-compatible schema
        cursor.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            stock INTEGER NOT NULL,
            date_added DATE NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS sales_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            sale_date DATE NOT NULL,
            quantity_sold INTEGER NOT NULL,
            revenue REAL NOT NULL,
            UNIQUE (product_id, sale_date)
        );

        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            training_date DATETIME,
            mae REAL,
            rmse REAL,
            r_squared REAL,
            status TEXT,
            file_path TEXT,
            product_id INTEGER
        );

        CREATE TABLE IF NOT EXISTS forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER,
            product_id INTEGER,
            forecast_date DATE,
            forecasted_quantity INTEGER,
            UNIQUE (model_id, product_id, forecast_date)
        );
        """)

        db.commit()

        # Seed users
        cursor.execute("SELECT COUNT(*) as count FROM users WHERE username = ?", ('admin@acme.com',))
        if cursor.fetchone()['count'] == 0:
            cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                           ('admin@acme.com', generate_password_hash('Admin@123'), 'Admin'))
            cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                           ('user@acme.com', generate_password_hash('User@123'), 'User'))
            db.commit()

def query_db(query, args=(), one=False):
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    cur.close()
    result = [dict(row) for row in rv]
    return (result[0] if result else None) if one else result

def execute_db(query, args=()):
    db = get_db()
    cur = db.execute(query, args)
    db.commit()
    return cur.lastrowid
