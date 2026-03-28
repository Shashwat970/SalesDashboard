"""
data_manager.py
---------------
All database read/write helpers used by app.py.
Works directly against the SQLite database via Flask's `g` / query_db helpers.
"""

from __future__ import annotations

import os
from datetime import datetime, date, timedelta
from typing import Any

from database import query_db, execute_db


class DataManager:
    """Thin service layer over the SQLite database."""

    # ──────────────────────────────────────────────────────────────────────────
    # Products
    # ──────────────────────────────────────────────────────────────────────────

    def get_all_products(self) -> list[dict]:
        return query_db("SELECT * FROM products ORDER BY name")

    def get_product_by_id(self, product_id: int) -> dict | None:
        return query_db("SELECT * FROM products WHERE id = ?", (product_id,), one=True)

    def get_product_categories(self) -> list[str]:
        rows = query_db("SELECT DISTINCT category FROM products ORDER BY category")
        return [r["category"] for r in rows]

    def add_product(self, data: dict) -> int:
        return execute_db(
            """INSERT INTO products (product_id, name, category, price, stock, date_added)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                data["product_id"],
                data["name"],
                data["category"],
                float(data["price"]),
                int(data["stock"]),
                data.get("date_added", datetime.now().strftime("%Y-%m-%d")),
            ),
        )

    def update_product(self, product_id: int, data: dict) -> None:
        execute_db(
            """UPDATE products
               SET name=?, category=?, price=?, stock=?, last_updated=?
               WHERE id=?""",
            (
                data.get("name"),
                data.get("category"),
                float(data.get("price", 0)),
                int(data.get("stock", 0)),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                product_id,
            ),
        )

    def delete_product(self, product_id: int) -> None:
        execute_db("DELETE FROM products WHERE id = ?", (product_id,))

    # ──────────────────────────────────────────────────────────────────────────
    # Sales data
    # ──────────────────────────────────────────────────────────────────────────

    def get_sales_data_for_product(self, product_id: int) -> list[dict]:
        return query_db(
            """SELECT sale_date, quantity_sold, revenue
               FROM sales_data
               WHERE product_id = ?
               ORDER BY sale_date""",
            (product_id,),
        )

    def get_recent_sales(self, limit: int = 10) -> list[dict]:
        return query_db(
            """SELECT s.sale_date, p.name AS product_name, s.quantity_sold, s.revenue
               FROM sales_data s
               JOIN products p ON p.id = s.product_id
               ORDER BY s.sale_date DESC
               LIMIT ?""",
            (limit,),
        )

    def upload_sales_data(self, df, uploaded_by: int = 1):
        """
        Insert/update rows from a pandas DataFrame.
        Expected columns: product_id, sale_date, quantity_sold, revenue
        Optional: promotion, holiday, weather
        Returns (inserted, updated, errors).
        """
        import pandas as pd

        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        required = {"product_id", "sale_date", "quantity_sold", "revenue"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {', '.join(missing)}")

        inserted = updated = 0
        errors: list[str] = []

        for idx, row in df.iterrows():
            try:
                pid_str = str(row["product_id"]).strip()
                product = query_db(
                    "SELECT id FROM products WHERE product_id = ?", (pid_str,), one=True
                )
                if not product:
                    errors.append(f"Row {idx+2}: product_id '{pid_str}' not found.")
                    continue

                db_pid    = product["id"]
                sale_date = str(row["sale_date"])[:10]
                qty       = int(row["quantity_sold"])
                revenue   = float(row["revenue"])

                existing = query_db(
                    "SELECT id FROM sales_data WHERE product_id=? AND sale_date=?",
                    (db_pid, sale_date),
                    one=True,
                )
                if existing:
                    execute_db(
                        "UPDATE sales_data SET quantity_sold=?, revenue=? WHERE id=?",
                        (qty, revenue, existing["id"]),
                    )
                    updated += 1
                else:
                    execute_db(
                        """INSERT INTO sales_data (product_id, sale_date, quantity_sold, revenue)
                           VALUES (?, ?, ?, ?)""",
                        (db_pid, sale_date, qty, revenue),
                    )
                    inserted += 1
            except Exception as exc:
                errors.append(f"Row {idx+2}: {exc}")

        return inserted, updated, errors

    # ──────────────────────────────────────────────────────────────────────────
    # Models / Forecasts
    # ──────────────────────────────────────────────────────────────────────────

    def get_all_models(self) -> list[dict]:
        return query_db("SELECT * FROM models ORDER BY training_date DESC")

    def get_model_by_id(self, model_id: int) -> dict | None:
        return query_db("SELECT * FROM models WHERE id = ?", (model_id,), one=True)

    def get_active_model(self, product_id: int | None = None) -> dict | None:
        if product_id is not None:
            m = query_db(
                "SELECT * FROM models WHERE status='active' AND (product_id=? OR product_id IS NULL) LIMIT 1",
                (product_id,),
                one=True,
            )
            if m:
                return m
        return query_db(
            "SELECT * FROM models WHERE status='active' ORDER BY training_date DESC LIMIT 1",
            one=True,
        )

    def set_active_model(self, model_id: int, product_id: int | None = None) -> None:
        if product_id:
            execute_db(
                "UPDATE models SET status='inactive' WHERE product_id=? OR product_id IS NULL",
                (product_id,),
            )
        else:
            execute_db("UPDATE models SET status='inactive'", ())
        execute_db("UPDATE models SET status='active' WHERE id=?", (model_id,))

    def delete_model(self, model_id: int) -> None:
        execute_db("DELETE FROM forecasts WHERE model_id=?", (model_id,))
        execute_db("DELETE FROM models WHERE id=?", (model_id,))

    def save_model_record(
        self,
        model_name: str,
        file_path: str,
        mae: float,
        rmse: float,
        r_squared: float,
        product_id: int | None = None,
        set_active: bool = True,
    ) -> int:
        """Persist a trained-model record to the DB and optionally activate it."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if set_active:
            execute_db("UPDATE models SET status='inactive'", ())
        model_id = execute_db(
            """INSERT INTO models
               (model_name, training_date, mae, rmse, r_squared, status, file_path, product_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                model_name,
                now,
                round(mae, 4),
                round(rmse, 4),
                round(r_squared, 4),
                "active" if set_active else "inactive",
                file_path,
                product_id,
            ),
        )
        return model_id

    def get_forecasts_for_product(self, product_id: int, model_id: int) -> list[dict]:
        return query_db(
            """SELECT forecast_date, forecasted_quantity
               FROM forecasts
               WHERE product_id=? AND model_id=?
               ORDER BY forecast_date""",
            (product_id, model_id),
        )

    def save_forecasts(
        self, model_id: int, product_id: int, forecasts: list[dict]
    ) -> None:
        """forecasts = [{'date': 'YYYY-MM-DD', 'quantity': int}, ...]"""
        execute_db(
            "DELETE FROM forecasts WHERE model_id=? AND product_id=?",
            (model_id, product_id),
        )
        for f in forecasts:
            try:
                execute_db(
                    """INSERT OR IGNORE INTO forecasts
                       (model_id, product_id, forecast_date, forecasted_quantity)
                       VALUES (?, ?, ?, ?)""",
                    (model_id, product_id, f["date"], int(f["quantity"])),
                )
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────────────────
    # Dashboard helpers
    # ──────────────────────────────────────────────────────────────────────────

    def get_dashboard_stats(self) -> dict:
        total_products = query_db("SELECT COUNT(*) as c FROM products",   one=True)["c"]
        total_users    = query_db("SELECT COUNT(*) as c FROM users",      one=True)["c"]
        total_models   = query_db("SELECT COUNT(*) as c FROM models",     one=True)["c"]
        last_upload    = query_db("SELECT MAX(sale_date) as d FROM sales_data", one=True)["d"]
        active_model   = self.get_active_model()
        return {
            "total_products":    total_products,
            "total_users":       total_users,
            "total_models":      total_models,
            "last_upload":       str(last_upload) if last_upload else "N/A",
            "active_model_name": active_model["model_name"] if active_model else "N/A",
            "active_model_date": str(active_model["training_date"]) if active_model else "N/A",
        }

    def get_dashboard_charts(self) -> dict:
        """Return accuracy data for the admin dashboard chart."""
        models = query_db(
            "SELECT training_date, r_squared FROM models ORDER BY training_date"
        )
        if models:
            labels = [m["training_date"][:7] for m in models]
            data   = [round(float(m["r_squared"]) * 100, 1) for m in models]
        else:
            # Stub data when no models exist yet
            labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
            data   = [88.0, 89.5, 91.0, 90.2, 92.4, 93.1]
        return {
            "accuracyChart": {
                "labels":   labels,
                "datasets": [{"label": "Accuracy (%)", "data": data}],
            }
        }

    def get_recent_uploads(self, limit: int = 10) -> list[dict]:
        """Best-effort: pull latest sale dates as a proxy for uploads."""
        rows = query_db(
            """SELECT p.name, s.sale_date as date
               FROM sales_data s
               JOIN products p ON p.id = s.product_id
               ORDER BY s.sale_date DESC
               LIMIT ?""",
            (limit,),
        )
        seen: dict[str, dict] = {}
        for r in rows:
            key = r["date"][:7]
            if key not in seen:
                seen[key] = {
                    "id":      len(seen) + 1,
                    "name":    f"sales_{key}.csv",
                    "records": 0,
                    "date":    r["date"] + " 00:00",
                }
            seen[key]["records"] += 1
        return list(seen.values())[:limit]

    def get_top_products(self, limit: int = 5) -> list[dict]:
        return query_db(
            """SELECT p.name, SUM(s.quantity_sold) as total_qty, SUM(s.revenue) as total_revenue
               FROM sales_data s JOIN products p ON p.id = s.product_id
               GROUP BY s.product_id ORDER BY total_revenue DESC LIMIT ?""",
            (limit,),
        )

    def get_category_breakdown(self) -> list[dict]:
        return query_db(
            """SELECT p.category, SUM(s.revenue) as total_revenue
               FROM sales_data s JOIN products p ON p.id = s.product_id
               GROUP BY p.category ORDER BY total_revenue DESC""",
        )

    def get_revenue_by_month(self) -> list[dict]:
        return query_db(
            """SELECT strftime('%Y-%m', sale_date) as month, SUM(revenue) as revenue
               FROM sales_data GROUP BY month ORDER BY month"""
        )
