from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import io

app = Flask(__name__)

# Load Model Artifacts
try:
    model = joblib.load('nids_model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')
    model_loaded = True
except Exception as e:
    model_loaded = False

# NSL-KDD / CICIDS2017 Multi-Class Attack Mapping
ATTACK_TYPES = {
    0: {'name': 'Normal', 'category': 'Safe', 'color': '#166534'},
    1: {'name': 'DoS (Denial of Service)', 'category': 'High Risk', 'color': '#991b1b'},
    2: {'name': 'Probe (Port Scan / Surveillance)', 'category': 'Medium Risk', 'color': '#d97706'},
    3: {'name': 'R2L (Remote to Local)', 'category': 'Critical Risk', 'color': '#7f1d1d'},
    4: {'name': 'U2R (User to Root)', 'category': 'Critical Risk', 'color': '#450a0a'}
}

@app.route('/')
def home():
    return render_template('index.html')

# --- SINGLE PACKET PREDICTION ---
@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({'status': 'error', 'message': 'Model files missing or not trained yet!'})

    try:
        data = request.json
        src_bytes = float(data.get('src_bytes', 0))
        dst_bytes = float(data.get('dst_bytes', 0))
        duration = float(data.get('duration', 0))

        # Artificial Buffer Overflow Anomaly Detection
        if src_bytes > 1000000 or dst_bytes > 1000000:
            return jsonify({
                'status': 'anomaly',
                'result': 'MALFORMED PACKET: Potential Buffer Overflow / Anomalous Transfer',
                'confidence': 99.9,
                'type': 'warning'
            })

        protocol = data.get('protocol', 'tcp')
        service = data.get('service', 'http')
        flag = data.get('flag', 'SF')

        # Feature Vector Formulation (41 features)
        features = np.zeros(41)
        features[0] = duration
        features[1] = encoders['protocol_type'].transform([protocol])[0] if protocol in encoders['protocol_type'].classes_ else 0
        features[2] = encoders['service'].transform([service])[0] if service in encoders['service'].classes_ else 0
        features[3] = encoders['flag'].transform([flag])[0] if flag in encoders['flag'].classes_ else 0
        features[4] = src_bytes
        features[5] = dst_bytes
        features[22] = float(data.get('count', 0))
        features[23] = float(data.get('srv_count', 0))
        features[28] = 1.0

        scaled = scaler.transform([features])
        pred_class = int(model.predict(scaled)[0])
        
        # Probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(scaled)[0]
            confidence = round(float(np.max(probs)) * 100, 2)
        else:
            confidence = 95.0

        attack_info = ATTACK_TYPES.get(pred_class, ATTACK_TYPES[1 if pred_class != 0 else 0])

        return jsonify({
            'status': 'success',
            'result': attack_info['name'],
            'category': attack_info['category'],
            'confidence': confidence,
            'is_attack': pred_class != 0
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


# --- BATCH CSV UPLOAD & ANALYTICS ---
@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded!'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'Empty file selected!'})

    try:
        df = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8")), header=None)
        
        if df.shape[1] >= 41:
            df_features = df.iloc[:, :41].copy()
        else:
            return jsonify({'status': 'error', 'message': f'CSV must contain at least 41 features. Found {df.shape[1]} columns.'})

        # Process Categoricals
        for idx, col_name in zip([1, 2, 3], ['protocol_type', 'service', 'flag']):
            le = encoders[col_name]
            df_features[idx] = df_features[idx].astype(str).map(
                lambda s: le.transform([s])[0] if s in le.classes_ else 0
            )

        df_scaled = scaler.transform(df_features.values)
        predictions = model.predict(df_scaled)

        # Calculate Analytics
        total_packets = len(predictions)
        normal_count = int(np.sum(predictions == 0))
        dos_count = int(np.sum(predictions == 1))
        probe_count = int(np.sum(predictions == 2))
        r2l_count = int(np.sum(predictions == 3))
        u2r_count = int(np.sum(predictions == 4))

        # Fallback if binary model is loaded
        if (dos_count + probe_count + r2l_count + u2r_count) == 0 and (total_packets - normal_count) > 0:
            dos_count = total_packets - normal_count

        return jsonify({
            'status': 'success',
            'total_packets': total_packets,
            'counts': {
                'Normal': normal_count,
                'DoS': dos_count,
                'Probe': probe_count,
                'R2L': r2l_count,
                'U2R': u2r_count
            }
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': f"CSV Processing Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
