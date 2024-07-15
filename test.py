import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from datetime import datetime, timedelta
import plotly.graph_objs as go
from flask import Flask, jsonify, render_template, request
import time
import threading
import random
import os

app = Flask(__name__)
model = load_model('LSTM.keras')
scaler = None
df_combined = pd.read_csv('output.csv')  # Initial combined data

# Initialize the start timestamp to the current time
start_timestamp = datetime.now()

# CSV file to store anomalies for each component
anomalies_csv = 'component_anomalies.csv'

# Create the CSV file if it doesn't exist
if not os.path.exists(anomalies_csv):
    df_components = pd.DataFrame(columns=['Compressor', 'Evaporator', 'Condenser'])
    df_components.to_csv(anomalies_csv, index=False)

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

def normalize_prediction_input(data, scaler):
    data_normalized = scaler.transform(data)
    return data_normalized

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
        time.sleep(.01)  # Add a new record every 5 seconds

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    global df_combined, scaler, model

    try:
        df_new = pd.read_csv('new_data.csv')
        if not df_new.empty:
            new_record = df_new.iloc[0:1]
            df_combined = pd.concat([df_combined, new_record]).reset_index(drop=True)
            df_new = df_new.iloc[1:]
            df_new.to_csv('new_data.csv', index=False)
    except pd.errors.EmptyDataError:
        df_new = pd.DataFrame()

    timestamps_combined = df_combined['Timestamp']
    data_combined = df_combined[['Probe_Temp']].values

    X_combined, y_combined, ts_combined = create_dataset(data_combined, timestamps_combined)
    temp_values = data_combined[:, 0]

    with open('threshold.txt', 'r') as f:
        temp_threshold = float(f.read())
    temp_highs = temp_values > temp_threshold
    temp_anomaly_timestamps = np.array(ts_combined)[temp_highs]

    sustained_anomalies = np.zeros(len(temp_highs), dtype=bool)
    window_size = 60
    tolerance = 5

    for i in range(len(temp_highs) - window_size + 1):
        if np.sum(temp_highs[i:i + window_size]) >= (window_size - tolerance):
            sustained_anomalies[i:i + window_size] = True

    faulty_component = "None"
    accuracy = 0.0

    if len(data_combined) == len(sustained_anomalies):
        X_sustained = data_combined[sustained_anomalies].reshape(-1, 1, 1)
        if X_sustained.size > 0:
            temp_diffs = X_sustained - temp_threshold
            X_sustained_diff = temp_diffs.reshape(-1, 1, 1)
            print(X_sustained_diff)

            predictions = model.predict(X_sustained_diff)
            print(f"Predictions shape: {predictions.shape}")
            print(f"Predictions: {predictions}")

            if predictions.shape[1] == 3:
                component_labels = ['compressor', 'evaporator', 'condenser']
                faulty_component_idx = np.argmax(predictions, axis=1)  # Get index of max value per sample
                print(f"Faulty component indices: {faulty_component_idx}")
                most_common_idx = np.bincount(faulty_component_idx).argmax()
                if 0 <= most_common_idx < len(component_labels):
                    faulty_component = component_labels[most_common_idx]
                    accuracy = np.mean(np.max(predictions, axis=1)) * 100
                    print(f"Predicted faulty component: {faulty_component} with accuracy {accuracy}%")
                else:
                    print(f"Invalid most_common_idx: {most_common_idx}, predictions: {predictions}")
            else:
                print("Predictions do not match expected shape for three classes.")
        else:
            print("Empty X_sustained array.")
    else:
        print("Mismatch in lengths between data_combined and sustained_anomalies.")

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

    df_combined['Timestamp'] = pd.to_datetime(df_combined['Timestamp'])

    # Find anomalies before the repair time
    anomalies_before_repair = df_combined[df_combined['Timestamp'] <= repair_time]
    anomalies_before_repair = anomalies_before_repair[repair_time - anomalies_before_repair['Timestamp'] <= timedelta(hours=1)]

    if not anomalies_before_repair.empty:
        print(f"Anomalies before repair:\n{anomalies_before_repair}")

        with open('threshold.txt', 'r') as f:
            temp_threshold = float(f.read())

        anomalies_data_diff = anomalies_before_repair[['Probe_Temp']].values - temp_threshold

        # Prepare data for training
        X_train = anomalies_data_diff[:, 0].reshape(-1, 1, 1)  # Adjusting shape for LSTM input
        print(X_train)
        component_label = components[0]  # Assuming the user selected component is the first element
        component_labels = {'compressor': 0, 'evaporator': 1, 'condenser': 2}
        y_train = np.full((X_train.shape[0],), component_labels[component_label])
        y_train = to_categorical(y_train, num_classes=3)  # One-hot encode the labels

        print(f"Training data (X_train) shape: {X_train.shape}")
        print(f"Training labels (y_train): {y_train}")

        # Train the model with the anomalies and their corresponding faulty component
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
        
        # Save updated model
        model.save('LSTM.keras')
        print('Supervised model trained and saved.')

    return jsonify({'status': 'success'})

if __name__ == '__main__':
    #data_thread = threading.Thread(target=add_new_data)
    #data_thread.start()
    app.run(debug=True)
