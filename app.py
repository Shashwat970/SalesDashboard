# app.py

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from config import Config
from database import init_db, close_db, query_db
from utils import data_manager, model_trainer
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__, static_url_path='/static')
app.config.from_object(Config)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize DB
with app.app_context():
    init_db(app)

app.teardown_appcontext(close_db)

# ---------------- AUTH ---------------- #

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first', 'warning')
            return redirect(url_for('login', next= request.url))
        return f(*args, **kwargs)
    return decorated


def role_required(role):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if 'user_id' not in session:
                return redirect(url_for('login', next= request.url))

            user = query_db("SELECT * FROM users WHERE id = ?", (session['user_id'],), one=True)

            if not user or user['role'] != role:
                flash('Unauthorized access', 'danger')
                return redirect(url_for('login'))

            return f(*args, **kwargs)
        return decorated
    return decorator


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# ---------------- ROUTES ---------------- #

@app.route('/')
def index():
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    # Only clear session after successful login (not on GET)
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = query_db("SELECT * FROM users WHERE username = ?", (username,), one=True)

        if user and check_password_hash(user['password'], password):
            session.clear()  # Reset session only after successful auth
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            flash('Welcome back!', 'success')
            next_page = request.args.get('next')
            if user['role'] == 'Admin':
                return redirect(next_page or url_for('admin_dashboard'))
            else:
                return redirect(next_page or url_for('forecast_dashboard'))
        else:
            # Generic error message for security
            flash('Invalid username or password.', 'danger')
            return render_template('login_page.html')

    # No session clearing on GET
    return render_template('login_page.html')



@app.route('/logout')
@login_required
def logout():
    session.clear()
    return redirect(url_for('login'))


# ---------------- DASHBOARDS ---------------- #

@app.route('/admin_dashboard')
@role_required('Admin')
def admin_dashboard():

    total_products = query_db("SELECT COUNT(*) as c FROM products", one=True)['c']
    total_users = query_db("SELECT COUNT(*) as c FROM users", one=True)['c']
    total_models = query_db("SELECT COUNT(*) as c FROM models", one=True)['c']

    active_model = data_manager.get_active_model()
    last_upload = query_db("SELECT MAX(sale_date) as d FROM sales_data", one=True)['d']

    return render_template(
        'admin_dashboard.html',
        total_products=total_products,
        total_users=total_users,
        total_models=total_models,
        active_model_name=active_model['model_name'] if active_model else "N/A",
        active_model_date=str(active_model['training_date']) if active_model else "N/A",
        last_upload=str(last_upload) if last_upload else "N/A",
        current_user=session.get('username')
    )


@app.route('/forecast_dashboard')
@login_required
def forecast_dashboard():
    return render_template('forecast_dashboard.html', current_user=session.get('username'))


# ---------------- PRODUCTS API ---------------- #

@app.route('/api/admin/products')
@role_required('Admin')
def get_products():
    products = data_manager.get_all_products()
    for p in products:
        p['price'] = float(p['price'])
    return jsonify(products)


# ---------------- UPLOAD ---------------- #

@app.route('/api/admin/uploads', methods=['POST'])
@role_required('Admin')
def upload_data():

    file = request.files.get('file')

    if not file:
        return jsonify({'error': 'No file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    try:
        df = pd.read_csv(path) if filename.endswith('.csv') else pd.read_excel(path)

        inserted, updated, errors = data_manager.upload_sales_data(
            df,
            session.get('user_id', 1)
        )

        return jsonify({
            'inserted': inserted,
            'updated': updated,
            'errors': errors
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------- FORECAST API ---------------- #

@app.route('/api/forecast/<int:product_id>')
@login_required
def forecast(product_id):

    product = data_manager.get_product_by_id(product_id)
    if not product:
        return jsonify({'error': 'Product not found'}), 404

    history = data_manager.get_sales_data_for_product(product_id)

    historical = [
        {'date': str(x['sale_date']), 'value': x['quantity_sold']}
        for x in history
    ]

    model = data_manager.get_active_model(product_id)
    if not model:
        return jsonify({'error': 'No model'}), 404

    forecasts = data_manager.get_forecasts_for_product(product_id, model['id'])

    if not forecasts:
        forecasts = model_trainer.generate_forecast(model['id'], product_id, 6)

    forecast_data = [
        {'date': str(x['forecast_date']), 'value': x['forecasted_quantity']}
        for x in forecasts
    ]

    return jsonify({
        'product': product['name'],
        'historical': historical,
        'forecast': forecast_data
    })


# ---------------- RUN (IMPORTANT) ---------------- #

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
