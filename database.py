# database.py
import mysql.connector
from mysql.connector import Error
from flask import current_app, g
from werkzeug.security import generate_password_hash
from flask import current_app as app

def get_db():
    """
    Establishes and returns a database connection and cursor.
    The connection is stored in Flask's `g` object for reuse within a request.
    """
    if 'db' not in g:
        try:
            g.db = mysql.connector.connect(
                host=current_app.config['MYSQL_HOST'],
                user=current_app.config['MYSQL_USER'],
                password=current_app.config['MYSQL_PASSWORD'],
                database=current_app.config['MYSQL_DB']
            )
            # Use dictionary=True to fetch rows as dictionaries, which is convenient for Flask's jsonify
            g.cursor = g.db.cursor(dictionary=True)
        except Error as e:
            print(f"Error connecting to MySQL database: {e}")
            raise # Re-raise the exception to propagate the error
    return g.db, g.cursor

def close_db(e=None):
    """
    Closes the database connection and cursor if they exist in Flask's `g` object.
    This function is registered as an app context teardown function.
    """
    db = g.pop('db', None)
    cursor = g.pop('cursor', None)
    if cursor is not None:
        cursor.close()
    if db is not None and db.is_connected():
        db.close()
        # print("MySQL connection closed.")

def init_db(app):
    """
    Initializes the database by creating tables and seeding initial users.
    This function is designed to be called once when the Flask app starts.
    """
    with app.app_context():
        db, cursor = get_db()
        try:
            # --- Table Creation ---
            # Create tables based on the schema defined in the project description
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(255) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL,
                    role ENUM('Admin', 'User') NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS products (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    product_id VARCHAR(50) UNIQUE NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    category VARCHAR(100) NOT NULL,
                    price DECIMAL(10, 2) NOT NULL,
                    stock INT NOT NULL,
                    date_added DATE NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sales_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    product_id INT NOT NULL,
                    sale_date DATE NOT NULL,
                    quantity_sold INT NOT NULL,
                    revenue DECIMAL(10, 2) NOT NULL,
                    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE,
                    UNIQUE (product_id, sale_date)
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    model_name VARCHAR(255) NOT NULL,
                    training_date DATETIME NOT NULL,
                    mae DECIMAL(10, 4),
                    rmse DECIMAL(10, 4),
                    r_squared DECIMAL(10, 4),
                    status ENUM('active', 'inactive', 'training', 'failed') NOT NULL,
                    file_path VARCHAR(255),
                    product_id INT NULL,
                    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE SET NULL
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS forecasts (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    model_id INT NOT NULL,
                    product_id INT NOT NULL,
                    forecast_date DATE NOT NULL,
                    forecasted_quantity INT NOT NULL,
                    FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE,
                    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE,
                    UNIQUE (model_id, product_id, forecast_date)
                );
            """)
            db.commit()
            print("Database tables checked/created successfully. ✨")

            # --- Seed Initial Users ---
            # Check if default admin user exists
            cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin@acme.com'")
            if cursor.fetchone()['COUNT(*)'] == 0:
                admin_pass_hash = generate_password_hash('Admin@123')
                user_pass_hash = generate_password_hash('User@123')
                manager_pass_hash = generate_password_hash('Manager@123')
                analyst_pass_hash = generate_password_hash('Analyst@123')

                cursor.execute("INSERT INTO users (username, password, role) VALUES (%s, %s, %s)",
                               ('admin@acme.com', admin_pass_hash, 'Admin'))
                cursor.execute("INSERT INTO users (username, password, role) VALUES (%s, %s, %s)",
                               ('user@acme.com', user_pass_hash, 'User'))
                db.commit()
                print("Default users added to database. 🧑‍💻")
        except Error as e:
            print(f"Error during database initialization: {e}")
            db.rollback() # Rollback any changes if an error occurs
        finally:
            close_db()

def query_db(query, args=(), one=False):
    """
    Executes a SELECT query and returns the results.
    :param query: The SQL query string.
    :param args: A tuple of arguments to be passed to the query.
    :param one: If True, returns only the first row; otherwise, returns all rows.
    :return: Query results.
    """
    db, cursor = get_db()
    cursor.execute(query, args)
    rv = cursor.fetchall()
    return (rv[0] if rv else None) if one else rv

def execute_db(query, args=()):
    """
    Executes an INSERT, UPDATE, or DELETE query and commits the changes.
    :param query: The SQL query string.
    :param args: A tuple of arguments to be passed to the query.
    :return: The last inserted row ID (if applicable).
    """
    db, cursor = get_db()
    try:
        cursor.execute(query, args)
        db.commit()
        return cursor.lastrowid
    except Error as e:
        db.rollback() # Rollback on error
        app.logger.error(f"Database execution error: {e}")
        raise # Re-raise the exception for upstream handling
