import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
import plotly.graph_objs as go
from flask import Flask, jsonify, render_template, request
import time
import threading
import random

app = Flask(__name__)
model = load_model('LSTM.keras')
scaler = None
df_combined = pd.read_csv('output.csv')  # Initial combined data

# Initialize the start timestamp to the current time
start_timestamp = datetime(year=2024, month=7, day=11, hour=11, minute=5, second=42)
start_timestamp = datetime.now()

def normalize_data(df, scaler=None):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Formatted_Timestamp'] = df['Timestamp'].dt.strftime('%m/%d/%Y %H:%M:%S')
    df = df.sort_values(by='Timestamp').reset_index(drop=True)
    data = df.drop(columns=['Timestamp', 'Formatted_Timestamp', 'File', 'Acceleration_X', 'Acceleration_Y', 
                            'Acceleration_Z', 'Altitude', 'Ambient_Temp', 'GPS_Fix', 'Humidity', 'Infrared', 
                            'Latitude', 'Light', 'Longitude', 'Magnetometer_X', 'Magnetometer_Y', 
                            'Magnetometer_Z', 'Visible'])
    timestamps = df['Formatted_Timestamp']
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)
    else:
        data = scaler.transform(data)
    return data, scaler, timestamps

def create_dataset(data, timestamps):
    X, y, ts = [], [], []
    for i in range(len(data)):
        X.append(data[i, :])
        y.append(data[i, :])
        ts.append(timestamps.iloc[i])
    return np.array(X), np.array(y), np.array(ts)

def add_new_data():
    global start_timestamp
    while True:
        new_timestamp = start_timestamp + timedelta(minutes=1)
        new_record = {
            'File': ['new_file'],
            'Timestamp': [new_timestamp.strftime('%m/%d/%Y %H:%M:%S.%f')],
            'Acceleration_X': [0], 'Acceleration_Y': [0], 'Acceleration_Z': [0], 'Altitude': [0],
            'Ambient_Temp': [0], 'GPS_Fix': [0], 'Humidity': [0], 'Infrared': [0], 'Latitude': [0],
            'Light': [0], 'Longitude': [0], 'Magnetometer_X': [0], 'Magnetometer_Y': [0], 'Magnetometer_Z': [0],
            'Probe_Temp': [round(random.uniform(20, 25), 2)], 'Visible': [0]
        }
        df_new = pd.DataFrame(new_record)
        df_new.to_csv('new_data.csv', mode='a', header=False, index=False)
        start_timestamp = new_timestamp  # Update the timestamp
          # Add a new record every 5 seconds

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    global df_combined, scaler, model

    # Simulate adding a new record from 'new_data.csv'
    try:
        df_new = pd.read_csv('new_data.csv')
        if not df_new.empty:
            new_record = df_new.iloc[0:1]
            df_combined = pd.concat([df_combined, new_record]).reset_index(drop=True)

            # Save the remaining new data
            df_new = df_new.iloc[1:]
            df_new.to_csv('new_data.csv', index=False)
    except pd.errors.EmptyDataError:
        # No new data, continue with existing data
        df_new = pd.DataFrame()

    # Normalize the combined data
    data_combined, scaler, timestamps_combined = normalize_data(df_combined, scaler)

    # Create dataset from combined data
    X_combined, y_combined, ts_combined = create_dataset(data_combined, timestamps_combined)

    y_inv = scaler.inverse_transform(y_combined)

    temp_values = y_inv[:, 0]

    # Detect anomalies based on actual Probe_Temp values
    with open('threshold.txt', 'r') as f:
        temp_threshold = float(f.read())

    # Normalize the temp_threshold
    temp_threshold_normalized = scaler.transform(np.array([np.full((data_combined.shape[1],), temp_threshold)]))[0, 0]

    temp_highs = data_combined[:, 0] > temp_threshold_normalized

    temp_anomaly_timestamps = np.array(ts_combined)[temp_highs]

    # Check for sustained anomalies in real time
    sustained_anomalies = np.zeros(len(temp_highs), dtype=bool)
    window_size = 60  # 60 minutes for one hour
    tolerance = 5  # Number of allowed interruptions

    for i in range(len(temp_highs) - window_size + 1):
        if np.sum(temp_highs[i:i + window_size]) >= (window_size - tolerance):
            sustained_anomalies[i:i + window_size] = True

    # Ensure the lengths match before reshaping
    if len(data_combined) == len(sustained_anomalies):
        X_sustained = data_combined[sustained_anomalies].reshape(-1, 1, data_combined.shape[1])
        if X_sustained.size > 0:
            temp_diffs = X_sustained - temp_threshold_normalized
            X_sustained_diff = temp_diffs.reshape(-1, 1, data_combined.shape[1])
            faulty_component_predictions = model.predict(X_sustained_diff)
            component_labels = ['compressor', 'evaporator', 'condenser']
            faulty_component_idx = np.argmax(faulty_component_predictions[0])
            faulty_component = component_labels[faulty_component_idx]
            accuracy = np.max(faulty_component_predictions[0]) * 100
        else:
            faulty_component = "None"
            accuracy = 0.0
    else:
        faulty_component = "None"
        accuracy = 0.0

    # Get timestamps for sustained anomalies
    sustained_anomaly_timestamps = np.array(ts_combined)[sustained_anomalies]

    return jsonify({
        'timestamps': timestamps_combined.tolist(),
        'y_inv': temp_values.tolist(),
        'temp_anomaly_timestamps': temp_anomaly_timestamps.tolist(),
        'y_temp_highs': temp_values[temp_highs].tolist(),
        'sustained_anomaly_timestamps': sustained_anomaly_timestamps.tolist(),
        'y_sustained_anomalies': temp_values[sustained_anomalies].tolist(),
        'temp_threshold': temp_threshold,
        'faulty_component': faulty_component,
        'accuracy': accuracy
    })


@app.route('/submit', methods=['POST'])
def submit():
    global model, scaler, df_combined
    data = request.get_json()
    repair_time = data['repair_time']
    components = data['components']

    # Process the repair time and components
    repair_time = datetime.strptime(repair_time, '%Y-%m-%d %H')
    print(f'Repair time: {repair_time}, Components: {components}')

    # Find anomalies before the repair time
    anomalies_before_repair = df_combined[df_combined['Timestamp'] <= repair_time]
    anomalies_before_repair = anomalies_before_repair[repair_time - anomalies_before_repair['Timestamp'] <= timedelta(hours=1)]

    if not anomalies_before_repair.empty:
        # Normalize the anomalies
        anomalies_data, _, _ = normalize_data(anomalies_before_repair, scaler)

        with open('threshold.txt', 'r') as f:
            temp_threshold = float(f.read())

        # Normalize the temp_threshold
        temp_threshold_normalized = scaler.transform(np.array([np.full((anomalies_data.shape[1],), temp_threshold)]))[0, 0]

        anomalies_data_diff = anomalies_data - temp_threshold_normalized

        # Save anomalies data to CSV
        df_anomalies = pd.DataFrame(anomalies_data_diff, columns=['Compressor', 'Evaporator', 'Condenser'])
        anomalies_data_diff.to_csv('anomalies.csv', index=False)

        # Read anomalies data from CSV
        df_anomalies = pd.read_csv('anomalies.csv')

        # Prepare data for training
        X_train = []
        y_train = []

        for component in components:
            values = df_anomalies[component].dropna().values
            labels = [component] * len(values)
            X_train.extend(values)
            y_train.extend(labels)

        # Convert lists to numpy arrays
        X_train = np.array(X_train).reshape(-1, 1)
        y_train = np.array(y_train)

        # Encode labels as integers
        component_labels = {'Compressor': 0, 'Evaporator': 1, 'Condenser': 2}
        y_train = np.array([component_labels[label] for label in y_train])

        # Reshape X_train to 3D array for LSTM input
        X_train = np.reshape(X_train, (X_train.shape[0], 1, 1))

        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

        # Convert X_anomalies and y_labels to DataFrame for viewing
        print("Saved anomalies:\n", df_anomalies)

    return jsonify({'status': 'success'})


if __name__ == '__main__':
    #data_thread = threading.Thread(target=add_new_data)
    #data_thread.start()
    app.run(debug=True)
