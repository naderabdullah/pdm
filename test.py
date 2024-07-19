import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras._tf_keras.keras.models import load_model, Sequential
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.layers import LSTM, Dropout, Dense, SimpleRNN
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.regularizers import l2
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
component_file = 'components.txt'
alert1 = False
alert2 = False
alert3 = False

# Initialize the start timestamp to the current time
start_timestamp = datetime.now()
start_timestamp = datetime(year=2024, month=7, day=19, hour=7, minute=55, second=35)

# CSV file to store anomalies for each component
anomalies_csv = 'component_anomalies.csv'

# Create the CSV file if it doesn't exist
if not os.path.exists(anomalies_csv):
    df_components = pd.DataFrame(columns=['Compressor', 'Evaporator', 'Condenser'])
    df_components.to_csv(anomalies_csv, index=False)

def expand_model(old_model, num_classes):
    # Extract weights from the existing model
    old_weights = [layer.get_weights() for layer in old_model.layers[:-1]]  # Exclude the output layer weights
    old_output_weights = old_model.layers[-1].get_weights()
    
    # Create a new model with the same architecture but expanded output layer
    new_model = Sequential()
    new_model.add(SimpleRNN(50, return_sequences=True, input_shape=(1, 1), kernel_regularizer=l2(0.001)))
    new_model.add(Dropout(0.2))
    new_model.add(SimpleRNN(50, return_sequences=False, kernel_regularizer=l2(0.001)))
    new_model.add(Dropout(0.2))
    new_model.add(Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.001)))
    
    # Set weights for the new model
    for i, layer in enumerate(new_model.layers[:-1]):
        layer.set_weights(old_weights[i])
    
    # Initialize new output layer weights
    old_output_weights[0] = np.pad(old_output_weights[0], ((0, 0), (0, num_classes - old_output_weights[0].shape[1])), 'constant')
    old_output_weights[1] = np.pad(old_output_weights[1], (0, num_classes - old_output_weights[1].shape[0]), 'constant')
    new_model.layers[-1].set_weights(old_output_weights)
    
    # Compile the new model
    new_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return new_model

def load_components():
    with open(component_file, 'r') as f:
        components = [line.strip() for line in f.readlines()]
    return components

def save_component(new_component):
    with open(component_file, 'a') as f:
        f.write(f"{new_component}\n")

# Method for normalizing the data, but normalization is not being used in training or prediction due to the existence of a threshold
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
            'Probe_Temp': [round(random.uniform(16, 17), 2)], 'Visible': [0], 'Voltage': [0]
        }
        df_new = pd.DataFrame(new_record)
        df_new.to_csv('new_data.csv', mode='a', header=False, index=False)
        start_timestamp = new_timestamp  # Update the timestamp
        time.sleep(.01)  # Add a new record every 5 seconds

def update_combined_data():
    global df_combined
    try:
        df_new = pd.read_csv('new_data.csv')
        if not df_new.empty:
            new_record = df_new.iloc[0:1]
            df_combined = pd.concat([df_combined, new_record]).reset_index(drop=True)
            df_new = df_new.iloc[1:]
            df_new.to_csv('new_data.csv', index=False)
    except pd.errors.EmptyDataError:
        df_new = pd.DataFrame()

@app.route('/')
def index():
    update_combined_data()
    return render_template('index.html')

@app.route('/data')
def data():
    global df_combined, model, alert1, alert2, alert3

    # Load the component list
    component_labels = load_components()

    update_combined_data()

    df_combined['Timestamp'] = pd.to_datetime(df_combined['Timestamp'])
    timestamps_combined = df_combined['Timestamp'].dt.strftime('%m/%d/%Y %H:%M:%S')
    data_combined = df_combined[['Probe_Temp']].values

    with open('threshold.txt', 'r') as f:
        temp_threshold = float(f.read())

    # Anomaly detection on the entire dataset
    temp_values = data_combined.flatten()
    temp_highs = temp_values > temp_threshold
    temp_anomaly_timestamps = np.array(timestamps_combined)[temp_highs]

    window_size = 60
    tolerance = 5
    sustained_anomalies = np.zeros_like(temp_highs, dtype=bool)

    for i in range(len(temp_highs) - window_size + 1):
        window_values = data_combined[i:i + window_size].flatten()
        if np.sum(temp_highs[i:i + window_size] & (window_values > temp_threshold)) >= (window_size - tolerance):
            sustained_anomalies[i:i + window_size] = (window_values > temp_threshold)

    sustained_anomaly_timestamps = np.array(timestamps_combined)[sustained_anomalies]

    # Prediction on the last hour of data
    last_day_df = df_combined[df_combined['Timestamp'] >= df_combined['Timestamp'].max() - timedelta(days=1)]
    last_day_data = last_day_df[['Probe_Temp']].values

    last_hour_df = df_combined[df_combined['Timestamp'] >= df_combined['Timestamp'].max() - timedelta(hours=1)]
    last_hour_timestamps = last_hour_df['Timestamp'].dt.strftime('%m/%d/%Y %H:%M:%S')
    last_hour_data = last_hour_df[['Probe_Temp']].values

    faulty_component = "None"
    accuracy = 0.0

    if not last_hour_df.empty and (last_hour_data > temp_threshold).any():
        temp_values_last_hour = last_hour_data.flatten()
        temp_highs_last_hour = temp_values_last_hour > temp_threshold

        if len(last_hour_data) == len(temp_highs_last_hour):
            X_sustained_last_hour = last_hour_data[temp_highs_last_hour].reshape(-1, 1, 1)
            print(X_sustained_last_hour.size)
            if X_sustained_last_hour.size == 30:
                alert1 = True
                alert2 = False
            elif X_sustained_last_hour.size == 60:
                alert2 = True
                alert1 = False
            else:
                alert1 = False
                alert2 = False
            if X_sustained_last_hour.size > 60:
                temp_diffs_last_hour = X_sustained_last_hour - temp_threshold
                X_sustained_diff_last_hour = temp_diffs_last_hour.reshape(-1, 1, 1)

                predictions = model.predict(X_sustained_diff_last_hour)
                print(f"Predictions shape: {predictions.shape}")
                print(f"Predictions: {predictions}")

                num_classes = predictions.shape[1]
                if num_classes > 0:
                    faulty_component_idx = np.argmax(predictions, axis=1)  # Get index of max value per sample
                    print(f"Faulty component indices: {faulty_component_idx}")
                    most_common_idx = np.bincount(faulty_component_idx).argmax()
                    if 0 <= most_common_idx < len(component_labels):
                        faulty_component = component_labels[most_common_idx]
                        accuracy = round(np.mean(np.max(predictions, axis=1)) * 100, 2)
                        if accuracy < 50:
                            faulty_component = "Unknown"
                            accuracy = 0.0
                        else:
                            print(f"Predicted faulty component: {faulty_component} with accuracy {accuracy}%")
                    else:
                        print(f"Invalid most_common_idx: {most_common_idx}, predictions: {predictions}")
                else:
                    print("Predictions do not match expected shape for any classes.")
            else:
                print("Empty X_sustained array.")
        else:
            print("Mismatch in lengths between data_combined and sustained_anomalies.")

    return jsonify({
        'timestamps': timestamps_combined.tolist(),
        'y_inv': temp_values.tolist(),
        'temp_anomaly_timestamps': temp_anomaly_timestamps.tolist(),
        'y_temp_highs': temp_values[temp_highs].tolist(),
        'sustained_anomaly_timestamps': sustained_anomaly_timestamps.tolist(),
        'y_sustained_anomalies': temp_values[sustained_anomalies].tolist(),
        'temp_threshold': temp_threshold,
        'faulty_component': faulty_component,
        'accuracy': accuracy,
        'alert1' : alert1,
        'alert2' : alert2,
        'alert3' : alert3
    })

@app.route('/get_components')
def get_components():
    update_combined_data()
    components = load_components()
    return jsonify({'components': components})

@app.route('/submit_form')
def submit_form():
    update_combined_data()
    return render_template('submit.html')

@app.route('/add_component', methods=['POST'])
def add_component():
    data = request.get_json()
    new_component = data.get('new_component', '').strip().lower()
    
    if new_component:
        component_labels = load_components()
        if new_component not in component_labels:
            save_component(new_component)
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'exists'})
    return jsonify({'status': 'error'})

@app.route('/submit', methods=['POST'])
def submit():
    global model, df_combined
    data = request.get_json()
    repair_time = data['repair_time']
    time_delta = int(data['time_delta'])
    components = data['components']
    new_component = data.get('new_component', '').strip().lower()

    # Load the component list
    component_labels = load_components()

    # Add new component if provided and not already in the list
    if new_component and new_component not in component_labels:
        component_labels.append(new_component)
        save_component(new_component)

    # Process the repair time and components
    repair_time = datetime.strptime(repair_time, '%m/%d/%Y %H')
    print(f'Repair time: {repair_time}, TimeDelta: {time_delta}, Components: {components}, New Component: {new_component}')

    df_combined['Timestamp'] = pd.to_datetime(df_combined['Timestamp'])

    # Find anomalies before the repair time
    anomalies_before_repair = df_combined[df_combined['Timestamp'] <= repair_time]
    anomalies_before_repair = anomalies_before_repair[repair_time - anomalies_before_repair['Timestamp'] <= timedelta(hours=time_delta)]

    if not anomalies_before_repair.empty:
        print(f"Anomalies before repair:\n{anomalies_before_repair}")

        with open('threshold.txt', 'r') as f:
            temp_threshold = float(f.read())

        anomalies_data_diff = anomalies_before_repair[['Probe_Temp']].values - temp_threshold

        avg = np.mean(anomalies_data_diff)

        avg_sample = np.array([[avg]])

        # Prepare data for training
        X_train = anomalies_data_diff.reshape(-1, 1)
        print(X_train)

        # Update component_labels dynamically
        component_labels_dict = {component: idx for idx, component in enumerate(component_labels)}
        if new_component and new_component not in component_labels:
            component_labels[new_component] = len(component_labels)

        # Assuming the user selected component is the first element
        component_label = components[0] if components else new_component
        y_train = np.full((X_train.shape[0],), component_labels_dict[component_label])
        y_train = to_categorical(y_train, num_classes=len(component_labels))  # One-hot encode the labels

        print(f"Training data (X_train) shape: {X_train.shape}")
        print(f"Training labels (y_train): {y_train}")

        # If the number of classes has changed, recreate and recompile the model
        if model.output_shape[1] != len(component_labels):
            model = expand_model(model, len(component_labels))
        
        # Train the model with the anomalies and their corresponding faulty component
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
        
        # Save updated model
        model.save('LSTM.keras')
        print('Supervised model trained and saved.')

    return jsonify({'status': 'success'})

if __name__ == '__main__':
    #data_thread = threading.Thread(target=add_new_data)
    #data_thread.start()
    app.run(debug=True)
