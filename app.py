# app.py
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, g, flash
from config import Config
from database import init_db, close_db, query_db, execute_db
from utils import data_manager, model_trainer
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import pandas as pd
from datetime import datetime
import os
import json

app = Flask(__name__, static_url_path='/static')
app.config.from_object(Config)

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database on app startup and register teardown function
with app.app_context():
    init_db(app)

app.teardown_appcontext(close_db)

# --- Authentication and Authorization Decorators ---
def login_required(f):
    """Decorator to ensure a user is logged in."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def role_required(role):
    """Decorator to restrict access based on user role."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                flash('Please log in to access this page.', 'warning')
                return redirect(url_for('login', next=request.url))
            user = query_db("SELECT * FROM users WHERE id = %s", (session['user_id'],), one=True)
            if not user or user['role'] != role:
                flash(f'You do not have permission to access this page. Required role: {role}.', 'danger')
                return redirect(url_for('login')) # Redirect to login or unauthorized page
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# --- Utility Functions ---
def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# --- Routes ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles user login and redirects based on role."""

    # Always reset session when visiting login page
    if request.method == 'GET':
        session.clear()

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = query_db("SELECT * FROM users WHERE username = %s", (username,), one=True)

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            flash(f'Welcome back, {session["username"]}!', 'success')
            next_page = request.args.get('next')
            if user['role'] == 'Admin':
                return redirect(next_page or url_for('admin_dashboard'))
            else:
                return redirect(next_page or url_for('forecast_dashboard'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')
            return render_template('login_page.html', error='Invalid username or password.')

    return render_template('login_page.html')


@app.route('/logout')
@login_required
def logout():
    """Logs out the current user by clearing all session data."""
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/')
def index():
    """Redirects to the login page as the default route."""
    return redirect(url_for('login'))

# --- Admin Module Routes ---

@app.route('/admin_dashboard')
@role_required('Admin')
def admin_dashboard():
    """Renders the Admin Dashboard."""
    # Fetch summary data for the dashboard
    total_products = query_db("SELECT COUNT(*) AS count FROM products", one=True)['count']
    total_users = query_db("SELECT COUNT(*) AS count FROM users", one=True)['count']
    active_model = data_manager.get_active_model()
    last_upload_data = query_db("SELECT MAX(sale_date) AS last_date FROM sales_data", one=True)['last_date']
    # The 'total_uploads' metric in the HTML is simplified to a count of models for demo
    total_models = query_db("SELECT COUNT(*) AS count FROM models", one=True)['count']

    active_model_info = {
        'name': active_model['model_name'] if active_model else "N/A",
        'date': active_model['training_date'].strftime('%Y-%m-%d') if active_model else "N/A"
    }

    return render_template('admin_dashboard.html',
                           total_products=total_products,
                           total_users=total_users,
                           active_model_info=active_model_info,
                           last_upload_date=last_upload_data.strftime('%Y-%m-%d') if last_upload_data else "N/A",
                           total_models=total_models,
                           current_user=session.get('username'))

@app.route('/forecast_dashboard')
def forecast_dashboard():
    """Renders the user-specific forecast dashboard page."""
    # Ensure logged in and role is valid
    if 'role' not in session or session['role'] not in ['User', 'Admin']:
        flash('Access denied. Please log in to view this page.', 'error')
        return redirect(url_for('login'))

    return render_template('forecast_dashboard.html', current_user=session.get('username'))

@app.route('/admin/products')
@role_required('Admin')
def product_management():
    """Renders the Product Management page."""
    categories = data_manager.get_product_categories()
    return render_template('product_management_page.html', categories=categories, current_user=session.get('username'))

@app.route('/admin/models')
@role_required('Admin')
def model_management():
    """Renders the Model Management page."""
    return render_template('model_management_page.html', current_user=session.get('username'))

@app.route('/admin/upload')
@role_required('Admin')
def upload_page():
    """Renders the Data Upload page."""
    return render_template('upload_page.html', current_user=session.get('username'))

@app.route('/admin/users')
@role_required('Admin')
def user_management():
    """Renders the User Management page."""
    return render_template('user_management_page.html', current_user=session.get('username'))

@app.route('/about.html')
def about_page():
    """Renders the About page."""
    return render_template('about.html', current_user=session.get('username'))

# --- API Endpoints (Admin) ---

@app.route('/api/admin/products', methods=['GET'])
@role_required('Admin')
def api_admin_get_products():
    """API to get all products."""
    products = data_manager.get_all_products()
    # Convert Decimal to float for JSON serialization
    for p in products:
        p['price'] = float(p['price'])
    return jsonify(products)

@app.route('/api/admin/products/<int:product_db_id>', methods=['GET'])
@role_required('Admin')
def api_admin_get_product(product_db_id):
    """API to get a single product by its database ID."""
    product = data_manager.get_product_by_id(product_db_id)
    if product:
        product['price'] = float(product['price'])
        return jsonify(product)
    return jsonify({'error': 'Product not found'}), 404

@app.route('/api/admin/products', methods=['POST'])
@role_required('Admin')
def api_admin_add_product():
    """API to add a new product."""
    data = request.get_json()
    # Basic validation
    if not all(key in data for key in ['product_id', 'name', 'category', 'price', 'stock']):
        return jsonify({'error': 'Missing required fields for product'}), 400
    try:
        data['date_added'] = datetime.now().strftime('%Y-%m-%d')
        new_id = data_manager.add_product(data)
        product = data_manager.get_product_by_id(new_id)
        if product:
            product['price'] = float(product['price'])
            return jsonify(product), 201
        return jsonify({'error': 'Failed to retrieve newly added product'}), 500
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@app.route('/api/admin/products/<int:product_db_id>', methods=['PUT'])
@role_required('Admin')
def api_admin_update_product(product_db_id):
    """API to update an existing product."""
    data = request.get_json()
    try:
        data_manager.update_product(product_db_id, data)
        product = data_manager.get_product_by_id(product_db_id)
        if product:
            product['price'] = float(product['price'])
            return jsonify(product)
        return jsonify({'error': 'Product not found after update'}), 404
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@app.route('/api/admin/products/<int:product_db_id>', methods=['DELETE'])
@role_required('Admin')
def api_admin_delete_product(product_db_id):
    """API to delete a product."""
    try:
        data_manager.delete_product(product_db_id)
        return jsonify({'message': 'Product deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@app.route('/api/admin/products/categories', methods=['GET'])
@role_required('Admin')
def api_admin_get_categories():
    """API to get all unique product categories."""
    categories = data_manager.get_product_categories()
    return jsonify(categories)

@app.route('/api/admin/models', methods=['GET'])
@role_required('Admin')
def api_admin_get_models():
    """API to get all trained models metadata."""
    models = data_manager.get_all_models()
    # Convert Decimal to float and datetime to string for JSON serialization
    for m in models:
        if m['mae'] is not None: m['mae'] = float(m['mae'])
        if m['rmse'] is not None: m['rmse'] = float(m['rmse'])
        if m['r_squared'] is not None: m['r_squared'] = float(m['r_squared'])
        m['training_date'] = m['training_date'].strftime('%Y-%m-%d %H:%M:%S')
    return jsonify(models)

@app.route('/api/admin/models/<int:model_db_id>/activate', methods=['POST'])
@role_required('Admin')
def api_admin_activate_model(model_db_id):
    """API to activate a specific model."""
    try:
        model = data_manager.get_model_by_id(model_db_id)
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        data_manager.set_active_model(model_db_id, product_id=model['product_id'])
        return jsonify({'message': f'Model {model_db_id} activated successfully!'}), 200
    except Exception as e:
        return jsonify({'error': f'Failed to activate model: {str(e)}'}), 500

@app.route('/api/admin/models/<int:model_db_id>', methods=['DELETE'])
@role_required('Admin')
def api_admin_delete_model(model_db_id):
    """API to delete a model and its associated file."""
    try:
        model = data_manager.get_model_by_id(model_db_id)
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        data_manager.delete_model(model_db_id)
        # Also delete the model file from disk
        if model['file_path'] and os.path.exists(model['file_path']):
            os.remove(model['file_path'])
        return jsonify({'message': 'Model deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': f'Failed to delete model: {str(e)}'}), 500

@app.route('/api/admin/models/train', methods=['POST'])
@role_required('Admin')
def api_admin_train_model():
    """API to initiate model training."""
    product_db_id = request.json.get('product_id') # Optional: train for specific product
    try:
        model_db_id, metrics = model_trainer.train_new_model(product_db_id)
        return jsonify({'message': 'Training initiated successfully!', 'model_id': model_db_id, 'metrics': metrics}), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        app.logger.error(f"Error during model training: {e}", exc_info=True)
        return jsonify({'error': f'Failed to train model: An unexpected error occurred. {str(e)}'}), 500

@app.route('/api/admin/uploads', methods=['POST'])
@role_required('Admin')
def api_admin_upload_data():
    """API to handle data file uploads (CSV/Excel)."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Read the file into a pandas DataFrame
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(filepath)
            else:
                return jsonify({'error': 'Unsupported file type'}), 400

            # Assuming the user is the one who uploaded the file, for simplicity
            uploaded_by_user_id = session.get('user_id', 1) # Fallback to admin ID if not set

            # Upload data to sales_data table and record upload history
            inserted, updated, errors = data_manager.upload_sales_data(df, uploaded_by_user_id, filename)

            if errors:
                return jsonify({'message': f'Uploaded {inserted} new records, updated {updated} existing records. Some errors occurred: {", ".join(errors)}', 'status': 'partial_success'}), 200
            return jsonify({'message': f'Uploaded {inserted} new records, updated {updated} existing records successfully!', 'status': 'success'}), 200
        except pd.errors.EmptyDataError:
            return jsonify({'error': 'The uploaded file is empty.'}), 400
        except pd.errors.ParserError as e:
            return jsonify({'error': f'Error parsing file. Check format: {str(e)}'}), 400
        except Exception as e:
            app.logger.error(f"Error processing upload: {e}", exc_info=True)
            return jsonify({'error': f'An unexpected error occurred during file processing: {str(e)}'}), 500
    else:
        return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/admin/dashboard_charts', methods=['GET'])
@role_required('Admin')
def api_admin_dashboard_charts():
    """
    API to provide data for the admin dashboard charts (Accuracy and Forecast vs Actual).
    This generates mock data for the demo, as real data would require complex aggregation
    and model evaluation history.
    """
    # Mock data for Forecast Accuracy (Last 12 Months)
    labels12 = ['Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']
    acc_data = [89.2, 88.7, 90.1, 89.4, 90.7, 91.3, 92.5, 91.8, 92.0, 92.7, 93.0, 92.1]
    accuracy_chart = {
        'labels': labels12,
        'datasets': [{'label': 'Accuracy %', 'data': acc_data}]
    }

    # Mock data for Forecast vs Actual (Last 6 Months)
    labels6 = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']
    forecast = [1210, 1312, 1254, 1320, 1410, 1505]
    actual = [1198, 1290, 1260, 1302, 1398, 1488]
    fva_chart = {
        'labels': labels6,
        'datasets': [
            {'label': 'Forecast', 'data': forecast},
            {'label': 'Actual', 'data': actual}
        ]
    }
    return jsonify({
        'accuracyChart': accuracy_chart,
        'fvaChart': fva_chart
    })

@app.route('/api/admin/latest_uploads', methods=['GET'])
@role_required('Admin')
def api_admin_get_latest_uploads():
    """API to get a summary of the latest data uploads."""
    uploads = data_manager.get_latest_uploads_summary(limit=10) # Using new data_manager function
    return jsonify(uploads)


@app.route('/api/admin/users', methods=['GET'])
@role_required('Admin')
def api_admin_get_users():
    """API to get all user accounts."""
    users = data_manager.get_users() # Using new data_manager function
    for user in users:
        user['created_at'] = user['created_at'].strftime('%Y-%m-%d %H:%M:%S')
    return jsonify(users)

@app.route('/api/admin/users/<int:user_id>', methods=['GET'])
@role_required('Admin')
def api_admin_get_user(user_id):
    """API to get a single user account by ID."""
    user = data_manager.get_user_by_id(user_id)
    if user:
        user['created_at'] = user['created_at'].strftime('%Y-%m-%d %H:%M:%S')
        return jsonify(user)
    return jsonify({'error': 'User not found'}), 404


@app.route('/api/admin/users', methods=['POST'])
@role_required('Admin')
def api_admin_add_user():
    """API to add a new user."""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    role = data.get('role')

    if not username or not password or not role:
        return jsonify({'error': 'Missing username, password, or role'}), 400

    if data_manager.get_user_by_username(username): # Check for username uniqueness
        return jsonify({'error': 'Username already exists'}), 409

    hashed_password = generate_password_hash(password)
    try:
        new_id = data_manager.add_user(username, hashed_password, role)
        user = data_manager.get_user_by_id(new_id)
        user['created_at'] = user['created_at'].strftime('%Y-%m-%d %H:%M:%S')
        return jsonify(user), 201
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@app.route('/api/admin/users/<int:user_id>', methods=['PUT'])
@role_required('Admin')
def api_admin_update_user(user_id):
    """API to update an existing user."""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password') # Optional: if not updating password
    role = data.get('role')

    # Check for username uniqueness if it's being changed
    if username:
        existing_user = data_manager.get_user_by_username(username)
        if existing_user and existing_user['id'] != user_id:
            return jsonify({'error': 'Username already exists for another user'}), 409

    hashed_password = generate_password_hash(password) if password else None
    try:
        data_manager.update_user(user_id, username, hashed_password, role)
        user = data_manager.get_user_by_id(user_id)
        user['created_at'] = user['created_at'].strftime('%Y-%m-%d %H:%M:%S')
        return jsonify(user), 200
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@app.route('/api/admin/users/<int:user_id>', methods=['DELETE'])
@role_required('Admin')
def api_admin_delete_user(user_id):
    """API to delete a user."""
    if user_id == session['user_id']:
        return jsonify({'error': 'Cannot delete your own active account.'}), 403
    try:
        data_manager.delete_user(user_id)
        return jsonify({'message': 'User deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500


# --- API Endpoints (User/Shared) ---

@app.route('/api/products', methods=['GET'])
@login_required
def api_get_products():
    """API to get all products (accessible by logged-in users)."""
    products = data_manager.get_all_products()
    for p in products:
        p['price'] = float(p['price'])
    return jsonify(products)

@app.route('/api/products/categories', methods=['GET'])
@login_required
def api_get_user_categories():
    """API to get all unique product categories (accessible by logged-in users)."""
    categories = data_manager.get_product_categories()
    return jsonify(categories)

@app.route('/api/forecast/<int:product_db_id>', methods=['GET'])
@login_required
def api_get_forecast_data(product_db_id):
    """
    API to get historical and forecasted data for a specific product.
    Automatically generates new forecasts if necessary using the active model.
    """
    try:
        product_info = data_manager.get_product_by_id(product_db_id)
        if not product_info:
            return jsonify({'error': 'Product not found'}), 404

        # Get historical data
        historical_sales = data_manager.get_sales_data_for_product(product_db_id)
        historical_data = [{
            'date': s['sale_date'].strftime('%Y-%m-%d'),
            'value': s['quantity_sold']
        } for s in historical_sales]

        # Get active model for this product (prioritizing product-specific, then global)
        active_model = data_manager.get_active_model(product_db_id)
        if not active_model:
            return jsonify({'error': 'No active model found for forecasting this product. Please ask an Admin to train and activate one.'}), 404

        forecast_data = []
        # Check if forecast already exists for this model and product (e.g., last 6 periods)
        existing_forecasts = data_manager.get_forecasts_for_product(product_db_id, active_model['id'])
        
        # If no existing forecasts or too few, generate new ones
        if not existing_forecasts or len(existing_forecasts) < 6:
            forecast_results = model_trainer.generate_forecast(active_model['id'], product_db_id, num_periods=6)
            forecast_data = [{
                'date': f['forecast_date'].strftime('%Y-%m-%d'),
                'value': f['forecasted_quantity']
            } for f in forecast_results]
        else:
            forecast_data = [{
                'date': f['forecast_date'].strftime('%Y-%m-%d'),
                'value': f['forecasted_quantity']
            } for f in existing_forecasts]

        response_data = {
            'product_name': product_info['name'],
            'historical_data': historical_data,
            'forecast_data': forecast_data,
            'active_model_info': {
                'id': active_model['id'],
                'name': active_model['model_name'],
                'mae': float(active_model['mae']) if active_model['mae'] else None
            }
        }
        return jsonify(response_data), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        app.logger.error(f"Error in api_get_forecast_data for product_id {product_db_id}: {e}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred during forecasting.'}), 500


if __name__ == '__main__':
    # Use app.run(debug=True) for development, turn off for production
    app.run(debug=True, host='0.0.0.0', port=5000)
