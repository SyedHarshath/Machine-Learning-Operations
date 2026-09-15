import base64
import io
import os
import time
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import OperationalError

app = Flask(__name__)

DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "heartdb")
DB_USER = os.getenv("DB_USER", "heartuser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "heartpass")

DATABASE_URL = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    "?charset=utf8mb4"
)
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=3600)

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

MODEL_CACHE = {"model": None, "features": FEATURES, "trained_at": None}


def wait_for_db(retries=30, delay=2):
    last_error = None
    for _ in range(retries):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return
        except OperationalError as exc:
            last_error = exc
            time.sleep(delay)
    raise last_error


def init_db():
    with engine.begin() as conn:
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS heart_records (
                id INT AUTO_INCREMENT PRIMARY KEY,
                age FLOAT NOT NULL,
                sex FLOAT NOT NULL,
                cp FLOAT NOT NULL,
                trestbps FLOAT NOT NULL,
                chol FLOAT NOT NULL,
                fbs FLOAT NOT NULL,
                restecg FLOAT NOT NULL,
                thalach FLOAT NOT NULL,
                exang FLOAT NOT NULL,
                oldpeak FLOAT NOT NULL,
                slope FLOAT NOT NULL,
                ca FLOAT NOT NULL,
                thal FLOAT NOT NULL,
                target INT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ))
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INT AUTO_INCREMENT PRIMARY KEY,
                model_name VARCHAR(100) NOT NULL,
                accuracy FLOAT NOT NULL,
                precision_score FLOAT NOT NULL,
                recall_score FLOAT NOT NULL,
                f1 FLOAT NOT NULL,
                roc_auc FLOAT NULL,
                confusion_matrix_json TEXT NOT NULL,
                classification_report TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ))
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS plots (
                id INT AUTO_INCREMENT PRIMARY KEY,
                plot_name VARCHAR(100) NOT NULL,
                image_data LONGBLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ))


def fetch_openml_heart():
    """Load the 303-row Heart-Disease dataset via scikit-learn/OpenML."""
    data = fetch_openml(name="heart-disease", version=1, as_frame=True)
    frame = data.frame.copy()

    # Convert all values to numeric; OpenML can expose categorical columns.
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    # The current dataset uses target as the label.
    if "target" not in frame.columns:
        target_candidates = [c for c in frame.columns if c.lower() in {"target", "class", "num"}]
        if not target_candidates:
            raise ValueError(f"Could not identify target column. Columns: {list(frame.columns)}")
        frame = frame.rename(columns={target_candidates[0]: "target"})

    missing_features = [c for c in FEATURES if c not in frame.columns]
    if missing_features:
        raise ValueError(f"Expected features missing from dataset: {missing_features}")

    frame = frame[FEATURES + ["target"]].dropna().reset_index(drop=True)
    # Heart disease source may represent target as 0/1 or numeric severity. Make it binary.
    frame["target"] = (frame["target"].astype(float) > 0).astype(int)
    return frame


def seed_dataset_if_empty():
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM heart_records")).scalar_one()
    if count:
        return count

    frame = fetch_openml_heart()
    rows = frame.to_dict(orient="records")
    insert_sql = text(
        """
        INSERT INTO heart_records
        (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target)
        VALUES
        (:age, :sex, :cp, :trestbps, :chol, :fbs, :restecg, :thalach, :exang, :oldpeak, :slope, :ca, :thal, :target)
        """
    )
    with engine.begin() as conn:
        conn.execute(insert_sql, rows)
    return len(rows)


def records_df():
    return pd.read_sql(text("SELECT age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target FROM heart_records ORDER BY id"), engine)


def build_model(model_name="Logistic Regression"):
    if model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42))
    ])


def save_plot(name, fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    data = buffer.getvalue()
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO plots (plot_name, image_data) VALUES (:name, :data)"), {"name": name, "data": data})
    plt.close(fig)
    return data


def train_and_store(model_name="Logistic Regression", test_size=0.20):
    df = records_df()
    if len(df) < 20:
        raise ValueError("At least 20 records are required to train the model.")

    X = df[FEATURES].astype(float)
    y = df["target"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    model = build_model(model_name)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    auc = roc_auc_score(y_test, proba) if proba is not None and len(np.unique(y_test)) == 2 else None
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, zero_division=0)

    with engine.begin() as conn:
        conn.execute(text(
            """
            INSERT INTO model_metrics
            (model_name, accuracy, precision_score, recall_score, f1, roc_auc, confusion_matrix_json, classification_report)
            VALUES (:model_name, :accuracy, :precision_score, :recall, :f1, :auc, :cm, :report)
            """
        ), {
            "model_name": model_name,
            "accuracy": float(accuracy),
            "precision_score": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc) if auc is not None else None,
            "cm": str(cm.tolist()),
            "report": report,
        })

    # Confusion matrix plot
    fig1, ax1 = plt.subplots(figsize=(5.2, 4.3))
    ax1.imshow(cm, interpolation="nearest")
    ax1.set_title(f"Confusion Matrix — {model_name}")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_xticks([0, 1], ["No Disease", "Disease"])
    ax1.set_yticks([0, 1], ["No Disease", "Disease"])
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, cm[i, j], ha="center", va="center")
    cm_data = save_plot("confusion_matrix", fig1)

    # ROC plot
    roc_data = None
    if proba is not None:
        fpr, tpr, _ = roc_curve(y_test, proba)
        fig2, ax2 = plt.subplots(figsize=(5.2, 4.3))
        ax2.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
        ax2.plot([0, 1], [0, 1], linestyle="--")
        ax2.set_title(f"ROC Curve — {model_name}")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.legend(loc="lower right")
        ax2.grid(alpha=0.25)
        roc_data = save_plot("roc_curve", fig2)

    MODEL_CACHE.update({"model": model, "features": FEATURES, "trained_at": datetime.now(timezone.utc).isoformat()})
    return {
        "model_name": model_name,
        "records": len(df),
        "test_records": len(X_test),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "plots_saved": ["confusion_matrix"] + (["roc_curve"] if roc_data else []),
        "trained_at": MODEL_CACHE["trained_at"],
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/summary")
def summary():
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM heart_records")).scalar_one()
        positive = conn.execute(text("SELECT COUNT(*) FROM heart_records WHERE target = 1")).scalar_one()
        negative = total - positive
        metric = conn.execute(text("SELECT model_name, accuracy, precision_score, recall_score, f1, roc_auc, created_at FROM model_metrics ORDER BY id DESC LIMIT 1")).mappings().first()
    return jsonify({
        "records": total,
        "positive": positive,
        "negative": negative,
        "latest_metrics": dict(metric) if metric else None,
    })


@app.get("/api/data")
def get_data():
    limit = min(int(request.args.get("limit", 100)), 500)
    offset = max(int(request.args.get("offset", 0)), 0)
    sql = text(
        "SELECT id, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target "
        "FROM heart_records ORDER BY id LIMIT :limit OFFSET :offset"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"limit": limit, "offset": offset}).mappings().all()
    return jsonify([dict(r) for r in rows])


@app.post("/api/data")
def create_record():
    payload = request.get_json(force=True)
    values = {field: float(payload[field]) for field in FEATURES}
    values["target"] = int(payload["target"])
    with engine.begin() as conn:
        result = conn.execute(text(
            "INSERT INTO heart_records (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target) "
            "VALUES (:age, :sex, :cp, :trestbps, :chol, :fbs, :restecg, :thalach, :exang, :oldpeak, :slope, :ca, :thal, :target)"
        ), values)
        new_id = result.lastrowid
    MODEL_CACHE["model"] = None
    return jsonify({"message": "Record created", "id": new_id}), 201


@app.put("/api/data/<int:record_id>")
def update_record(record_id):
    payload = request.get_json(force=True)
    values = {field: float(payload[field]) for field in FEATURES}
    values["target"] = int(payload["target"])
    values["id"] = record_id
    with engine.begin() as conn:
        result = conn.execute(text(
            "UPDATE heart_records SET age=:age, sex=:sex, cp=:cp, trestbps=:trestbps, chol=:chol, fbs=:fbs, restecg=:restecg, "
            "thalach=:thalach, exang=:exang, oldpeak=:oldpeak, slope=:slope, ca=:ca, thal=:thal, target=:target WHERE id=:id"
        ), values)
        if result.rowcount == 0:
            return jsonify({"error": "Record not found"}), 404
    MODEL_CACHE["model"] = None
    return jsonify({"message": "Record updated"})


@app.delete("/api/data/<int:record_id>")
def delete_record(record_id):
    with engine.begin() as conn:
        result = conn.execute(text("DELETE FROM heart_records WHERE id=:id"), {"id": record_id})
        if result.rowcount == 0:
            return jsonify({"error": "Record not found"}), 404
    MODEL_CACHE["model"] = None
    return jsonify({"message": "Record deleted"})


@app.post("/api/train")
def train():
    payload = request.get_json(silent=True) or {}
    model_name = payload.get("model_name", "Logistic Regression")
    if model_name not in {"Logistic Regression", "Random Forest"}:
        return jsonify({"error": "Unsupported model"}), 400
    try:
        return jsonify(train_and_store(model_name))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/metrics")
def metrics():
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT id, model_name, accuracy, precision_score, recall_score, f1, roc_auc, confusion_matrix_json, classification_report, created_at "
            "FROM model_metrics ORDER BY id DESC LIMIT 1"
        )).mappings().first()
    return jsonify(dict(row) if row else {})


@app.get("/api/plot/<plot_name>")
def latest_plot(plot_name):
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT image_data FROM plots WHERE plot_name=:name ORDER BY id DESC LIMIT 1"
        ), {"name": plot_name}).first()
    if not row:
        return jsonify({"error": "Plot not found"}), 404
    return send_file(io.BytesIO(row[0]), mimetype="image/png", download_name=f"{plot_name}.png")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    wait_for_db()
    init_db()
    try:
        seed_dataset_if_empty()
    except Exception as exc:
        # Keep the web app reachable even when OpenML is temporarily unavailable.
        app.logger.warning("Dataset seed skipped: %s", exc)
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    app.run(host=host, port=port)
