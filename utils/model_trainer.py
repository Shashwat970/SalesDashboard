"""
model_trainer.py
----------------
Trains scikit-learn regression models on sales data and persists them with
joblib. Also handles forecast generation from saved models.
"""

from __future__ import annotations

import os
import math
import pickle
from datetime import datetime, timedelta, date
from typing import Any

import numpy as np

# ── Lazy sklearn imports so the app still starts if sklearn is missing ─────────
def _sklearn_imports():
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return (
        GradientBoostingRegressor, RandomForestRegressor, Ridge,
        StandardScaler, Pipeline,
        cross_val_score,
        mean_absolute_error, mean_squared_error, r2_score,
    )


# ── Try joblib, fall back to pickle ───────────────────────────────────────────
try:
    import joblib
    _save  = joblib.dump
    _load  = joblib.load
    _EXT   = ".joblib"
except ImportError:
    import pickle as joblib                         # type: ignore[no-redef]
    def _save(obj, path):                           # type: ignore[misc]
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    def _load(path):                                # type: ignore[misc]
        with open(path, "rb") as f:
            return pickle.load(f)
    _EXT = ".pkl"


# ─────────────────────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Trains a GradientBoostingRegressor (with Ridge fallback) on historical
    sales data, saves the fitted pipeline to disk, and records metadata in
    the database.
    """

    def __init__(self, model_save_dir: str = "models"):
        self.model_save_dir = model_save_dir
        os.makedirs(model_save_dir, exist_ok=True)

    # ─── Feature Engineering ─────────────────────────────────────────────────

    @staticmethod
    def _build_features(dates: list[str]) -> np.ndarray:
        """
        Turn a list of 'YYYY-MM-DD' strings into a numeric feature matrix:
        [year, month, day_of_year, quarter, month_sin, month_cos,
         days_since_start, week_of_year]
        """
        rows = []
        for d_str in dates:
            try:
                d = datetime.strptime(str(d_str)[:10], "%Y-%m-%d")
            except ValueError:
                d = datetime.now()
            doy  = d.timetuple().tm_yday
            rows.append([
                d.year,
                d.month,
                doy,
                (d.month - 1) // 3 + 1,                     # quarter 1-4
                math.sin(2 * math.pi * d.month / 12),        # seasonality
                math.cos(2 * math.pi * d.month / 12),
                (d - datetime(2020, 1, 1)).days,              # trend proxy
                d.isocalendar()[1],                           # ISO week
            ])
        return np.array(rows, dtype=float)

    @staticmethod
    def _future_dates(n: int = 6) -> list[str]:
        """Return n monthly date strings starting next month."""
        today = date.today()
        results = []
        y, m = today.year, today.month
        for _ in range(n):
            m += 1
            if m > 12:
                m = 1
                y += 1
            results.append(f"{y}-{m:02d}-01")
        return results

    # ─── Training ────────────────────────────────────────────────────────────

    def train_new_model(
        self,
        product_id: int | None = None,
        n_forecast_months: int = 6,
    ) -> tuple[int, dict]:
        """
        Train a model on available sales data.

        Parameters
        ----------
        product_id : int | None
            If given, train only on that product's data.
            If None, train a global model across all products.
        n_forecast_months : int
            How many future monthly periods to pre-generate.

        Returns
        -------
        (db_model_id, metrics_dict)
        """
        # Import sklearn lazily
        (
            GradientBoostingRegressor, RandomForestRegressor, Ridge,
            StandardScaler, Pipeline,
            cross_val_score,
            mean_absolute_error, mean_squared_error, r2_score,
        ) = _sklearn_imports()

        from database import query_db
        from utils.data_manager import DataManager
        dm = DataManager()

        # ── Load training data ──────────────────────────────────────────────
        if product_id:
            rows = query_db(
                "SELECT sale_date, quantity_sold FROM sales_data WHERE product_id=? ORDER BY sale_date",
                (product_id,),
            )
        else:
            rows = query_db(
                "SELECT sale_date, quantity_sold FROM sales_data ORDER BY sale_date"
            )

        if not rows or len(rows) < 4:
            raise ValueError(
                "Not enough sales data to train (need at least 4 records). "
                "Please upload data first."
            )

        dates  = [r["sale_date"] for r in rows]
        y_vals = np.array([float(r["quantity_sold"]) for r in rows], dtype=float)
        X      = self._build_features(dates)

        # ── Build & fit pipeline ────────────────────────────────────────────
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                random_state=42,
            )),
        ])

        # Cross-val if enough data, else just fit
        if len(rows) >= 10:
            cv_scores = cross_val_score(
                pipeline, X, y_vals,
                cv=min(5, len(rows) // 2),
                scoring="r2",
            )
            pipeline.fit(X, y_vals)
            y_pred   = pipeline.predict(X)
            r_squared = float(np.mean(cv_scores))           # CV R² is more honest
        else:
            pipeline.fit(X, y_vals)
            y_pred    = pipeline.predict(X)
            ss_res    = np.sum((y_vals - y_pred) ** 2)
            ss_tot    = np.sum((y_vals - np.mean(y_vals)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot else 0.0

        mae  = float(mean_absolute_error(y_vals, y_pred))
        rmse = float(math.sqrt(mean_squared_error(y_vals, y_pred)))
        r_squared = max(-1.0, min(1.0, r_squared))          # clamp

        # ── Persist model file ──────────────────────────────────────────────
        ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
        scope      = f"p{product_id}" if product_id else "global"
        filename   = f"model_{scope}_{ts}{_EXT}"
        file_path  = os.path.join(self.model_save_dir, filename)
        model_name = f"GBR_{scope}_{ts}"

        _save(pipeline, file_path)

        # ── Save record to DB (and activate) ───────────────────────────────
        db_model_id = dm.save_model_record(
            model_name  = model_name,
            file_path   = file_path,
            mae         = mae,
            rmse        = rmse,
            r_squared   = r_squared,
            product_id  = product_id,
            set_active  = True,
        )

        # ── Pre-generate forecasts ──────────────────────────────────────────
        future_dates = self._future_dates(n_forecast_months)
        X_future     = self._build_features(future_dates)
        y_future     = pipeline.predict(X_future)

        # If product-specific, save for that product only
        # If global, save for every product
        if product_id:
            target_products = [product_id]
        else:
            products = query_db("SELECT id FROM products")
            target_products = [p["id"] for p in products]

        for pid in target_products:
            forecasts = [
                {"date": d, "quantity": max(0, round(float(q)))}
                for d, q in zip(future_dates, y_future)
            ]
            dm.save_forecasts(db_model_id, pid, forecasts)

        metrics = {
            "mae":       round(mae, 4),
            "rmse":      round(rmse, 4),
            "r_squared": round(r_squared, 4),
            "model_name": model_name,
            "file_path":  file_path,
            "records_used": len(rows),
        }
        return db_model_id, metrics

    # ─── Forecast generation from a saved model ───────────────────────────────

    def generate_forecast(
        self,
        model_id: int,
        product_id: int,
        n_months: int = 6,
    ) -> list[dict]:
        """
        Load a saved model from disk and generate forecasts for a product.
        Returns a list of dicts matching the `forecasts` table schema.
        """
        from database import query_db
        from utils.data_manager import DataManager
        dm = DataManager()

        model_row = query_db("SELECT * FROM models WHERE id=?", (model_id,), one=True)
        if not model_row:
            raise ValueError(f"Model {model_id} not found in database.")

        file_path = model_row.get("file_path", "")
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Model file not found at '{file_path}'. "
                "Please re-train the model."
            )

        pipeline     = _load(file_path)
        future_dates = self._future_dates(n_months)
        X_future     = self._build_features(future_dates)
        y_future     = pipeline.predict(X_future)

        forecasts = [
            {"date": d, "quantity": max(0, round(float(q)))}
            for d, q in zip(future_dates, y_future)
        ]
        dm.save_forecasts(model_id, product_id, forecasts)

        # Return in the shape the route expects
        return [
            {"forecast_date": f["date"], "forecasted_quantity": f["quantity"]}
            for f in forecasts
        ]

    # ─── Convenience: retrain & activate ─────────────────────────────────────

    def retrain_and_activate(
        self,
        product_id: int | None = None,
        n_forecast_months: int = 6,
    ) -> tuple[int, dict]:
        """Alias kept for compatibility."""
        return self.train_new_model(product_id, n_forecast_months)
