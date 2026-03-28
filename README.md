# ACME Sales Forecasting System

A Flask-based sales forecasting web application with a clean minimalist UI.

## Project Structure

```
SalesForecast/
├── app.py                  # Flask application & all routes
├── config.py               # Configuration
├── database.py             # SQLite setup & helpers
├── requirements.txt
├── uploads/                # Uploaded CSV/Excel files (auto-created)
├── models/                 # Saved ML model files (auto-created)
├── static/
│   ├── css/
│   │   ├── style.css                   # Global design system
│   │   ├── login_page.css
│   │   ├── admin_dashboard.css
│   │   ├── forecast_dashboard.css
│   │   ├── product_management_page.css
│   │   ├── model_management_page.css
│   │   ├── upload_page.css
│   │   └── user_management_page.css
│   └── img/
│       └── logo.jpg
└── templates/
    ├── base.html                   # Shared layout (navbar + sidebar)
    ├── login_page.html
    ├── admin_dashboard.html
    ├── forecast_dashboard.html
    ├── products_page.html
    ├── model_management_page.html
    ├── upload_page.html
    └── user_management_page.html
```

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run
python app.py
```

Open http://localhost:5000 in your browser.

## Default Credentials

| Role  | Username         | Password   |
|-------|-----------------|------------|
| Admin | admin@acme.com  | Admin@123  |
| User  | user@acme.com   | User@123   |

## Pages

| Page                  | URL          | Role  |
|-----------------------|-------------|-------|
| Login                 | /login      | All   |
| Admin Dashboard       | /admin      | Admin |
| Sales Forecasts       | /forecast   | All   |
| Product Management    | /products   | Admin |
| Upload Data           | /upload     | Admin |
| Model Management      | /models     | Admin |
| User Management       | /users      | Admin |

## Design

- **Font**: DM Sans + DM Mono
- **Theme**: Clean minimalist — white surfaces, neutral grays, dark navy primary
- **Accent**: Crimson red (`#e94560`)
- All pages share a fixed top navbar + collapsible sidebar via `base.html`
- CSS uses custom properties (CSS variables) for consistent theming

## Adding ML Utils

Create `utils/data_manager.py` and `utils/model_trainer.py` with the functions
the app expects. The app gracefully falls back to demo data if these modules
are missing.
