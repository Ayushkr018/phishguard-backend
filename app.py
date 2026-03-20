from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from xgboost import XGBClassifier
import json, os, warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# TRAIN MODELS ON STARTUP
# ─────────────────────────────────────────────
print("Loading datasets...")
train_df = pd.read_csv('Phising_Training_Dataset.csv')
test_df  = pd.read_csv('Phising_Testing_Dataset.csv')

FEATURES     = [c for c in train_df.columns if c not in ['key', 'Result']]
X_train      = train_df[FEATURES]
y_train      = train_df['Result']
y_map        = y_train.map({1: 1, -1: 0})

print("Training models...")

# Feature importance via Random Forest
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
SORTED_IMP   = sorted(
    zip(FEATURES, [round(float(x), 6) for x in rf.feature_importances_]),
    key=lambda x: x[1], reverse=True
)
TOP2         = [SORTED_IMP[0][0], SORTED_IMP[1][0]]
FEATURES_RED = [f for f in FEATURES if f not in TOP2]

# Full model — 30 features
XGB_FULL = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='logloss', random_state=42, n_jobs=-1
)
XGB_FULL.fit(X_train, y_map)

# Reduced model — 28 features (top 2 removed)
XGB_RED = XGBClassifier(
    n_estimators=400, max_depth=7, learning_rate=0.08,
    subsample=0.85, colsample_bytree=0.75,
    min_child_weight=2, gamma=0.1,
    eval_metric='logloss', random_state=42, n_jobs=-1
)
XGB_RED.fit(train_df[FEATURES_RED], y_map)

# Load precomputed stats (accuracy, metrics, etc.)
with open('stats.json') as f:
    STATS = json.load(f)

print(f"Ready! Full: {STATS['full_model']['accuracy']}% | Reduced: {STATS['reduced_model']['accuracy']}%")


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "PhishGuard API is running",
        "version": "1.0",
        "endpoints": [
            "GET  /api/stats",
            "GET  /api/features",
            "POST /api/predict/single",
            "POST /api/predict/batch"
        ]
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """All model stats, metrics, feature importances for dashboard"""
    return jsonify(STATS)


@app.route('/api/features', methods=['GET'])
def get_features():
    """Feature list with importance scores"""
    return jsonify({
        "all_features": FEATURES,
        "reduced_features": FEATURES_RED,
        "removed": TOP2,
        "importance": SORTED_IMP
    })


@app.route('/api/predict/single', methods=['POST'])
def predict_single():
    """
    Predict one website.
    Body: JSON with feature values + optional "model": "full" or "reduced"
    Example: { "SSLfinal_State": -1, "URL_of_Anchor": -1, ..., "model": "full" }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Send JSON body"}), 400

        model_type      = data.pop('model', 'full')
        features_to_use = FEATURES_RED if model_type == 'reduced' else FEATURES
        model           = XGB_RED      if model_type == 'reduced' else XGB_FULL

        # Build row — default 0 for missing features
        row      = {f: int(data.get(f, 0)) for f in features_to_use}
        pred_raw = model.predict(pd.DataFrame([row]))[0]
        pred     = 1 if pred_raw == 1 else -1

        suspicious = [f for f in features_to_use if row[f] == -1]
        safe       = [f for f in features_to_use if row[f] == 1]

        return jsonify({
            "prediction": pred,
            "label": "Legitimate" if pred == 1 else "Phishing",
            "model_used": model_type,
            "suspicious_features": suspicious,
            "safe_features": safe,
            "suspicious_count": len(suspicious),
            "total_features": len(features_to_use)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict multiple websites via CSV upload or JSON array.
    Query param: ?model=full or ?model=reduced
    """
    try:
        model_type = request.args.get('model', 'full')

        if 'file' in request.files:
            df = pd.read_csv(request.files['file'])
        elif request.is_json:
            data = request.get_json()
            df   = pd.DataFrame(data if isinstance(data, list) else [data])
        else:
            return jsonify({"error": "Send a CSV file or JSON array"}), 400

        features_to_use = FEATURES_RED if model_type == 'reduced' else FEATURES
        model           = XGB_RED      if model_type == 'reduced' else XGB_FULL
        has_key         = 'key' in df.columns

        for f in features_to_use:
            if f not in df.columns:
                df[f] = 0

        preds = [1 if p == 1 else -1 for p in model.predict(df[features_to_use])]

        results = []
        for i, pred in enumerate(preds):
            row = {
                "index": i,
                "prediction": pred,
                "label": "Legitimate" if pred == 1 else "Phishing"
            }
            if has_key:
                row["key"] = int(df['key'].iloc[i])
            results.append(row)

        total    = len(preds)
        phishing = preds.count(-1)
        legit    = preds.count(1)

        return jsonify({
            "summary": {
                "total": total,
                "phishing": phishing,
                "legitimate": legit,
                "phishing_pct": round(phishing / total * 100, 1),
                "legitimate_pct": round(legit / total * 100, 1),
                "model_used": model_type
            },
            "predictions": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
