import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from tensorflow import keras
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dropout, Dense
from keras._tf_keras.keras.regularizers import l2
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.utils import to_categorical
import random
from sensor_module import initialize_sensors
import adafruit_max31865
from adafruit_ina219 import ADCResolution, BusVoltageRange, INA219

component_file = 'components.txt'
scaler = None

# Initialize sensors
spi, cs, max31865, i2c_bus, ina219 = initialize_sensors()

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

def update_combined_data():
    try:
        df_new = pd.read_csv('new_data.csv')
        if not df_new.empty:
            new_record = df_new.iloc[0:1]
            df_combined = pd.concat([df_new, new_record]).reset_index(drop=True)
            df_new = df_new.iloc[1:]
            df_new.to_csv('new_data.csv', index=False)
    except pd.errors.EmptyDataError:
        df_new = pd.DataFrame()

def load_components():
    with open(component_file, 'r') as f:
        components = [line.strip() for line in f.readlines()]
    return components

def save_component(new_component):
    with open(component_file, 'a') as f:
        f.write(f"{new_component}\n")

def expand_model(old_model, num_classes):
    old_weights = [layer.get_weights() for layer in old_model.layers[:-1]]
    old_output_weights = old_model.layers[-1].get_weights()
    
    new_model = Sequential()
    new_model.add(LSTM(50, return_sequences=True, input_shape=(1, 3), kernel_regularizer=l2(0.01)))
    new_model.add(Dropout(0.2))
    new_model.add(LSTM(50, return_sequences=False, kernel_regularizer=l2(0.01)))
    new_model.add(Dropout(0.2))
    new_model.add(Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01)))
    
    for i, layer in enumerate(new_model.layers[:-1]):
        layer.set_weights(old_weights[i])
    
    old_output_weights[0] = np.pad(old_output_weights[0], ((0, 0), (0, num_classes - old_output_weights[0].shape[1])), 'constant')
    old_output_weights[1] = np.pad(old_output_weights[1], (0, num_classes - old_output_weights[1].shape[0]), 'constant')
    new_model.layers[-1].set_weights(old_output_weights)
    
    new_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return new_model

def add_new_data(time):
    new_timestamp = time + timedelta(minutes=1)
    new_record = {
        'File': ['new_file'],
        'Timestamp': [new_timestamp.strftime('%m/%d/%Y %H:%M:%S.%f')],
        'Acceleration_X': [0], 'Acceleration_Y': [0], 'Acceleration_Z': [0], 'Altitude': [0],
        'Ambient_Temp': [0], 'GPS_Fix': [0], 'Humidity': [0], 'Infrared': [0], 'Latitude': [0],
        'Light': [0], 'Longitude': [0], 'Magnetometer_X': [0], 'Magnetometer_Y': [0], 'Magnetometer_Z': [0],
        'Probe_Temp': [round(max31865.temperature * 9 / 5 + 32, 3)], 'Visible': [0], 
        'Board_voltage': [round(ina219.bus_voltage + ina219.shunt_voltage, 3)], 'Compressor_Voltage': [round(random.uniform(119, 121), 3)]
    }

    df_new = pd.DataFrame(new_record)
    df_new.to_csv('new_data.csv', mode='a', header=False, index=False)
    start_timestamp = new_timestamp

def prepare_training_data(anomalies_before_repair, component_labels, new_component, components):
    with open('threshold.txt', 'r') as f:
        temp_threshold = float(f.read())  

    board_voltage_threshold = 4.65
    voltage_threshold = 115

    anomalies_data_diff = anomalies_before_repair[['Probe_Temp', 'Board_Voltage', 'Compressor_Voltage']].values - [temp_threshold, board_voltage_threshold, voltage_threshold]
    avg = np.mean(anomalies_data_diff)
    avg_sample = np.array([[avg]])

    X_train = anomalies_data_diff.reshape(-1, 1, 3)

    component_labels_dict = {component: idx for idx, component in enumerate(component_labels)}
    if new_component and new_component not in component_labels:
        component_labels[new_component] = len(component_labels)

    component_label = components[0] if components else new_component
    y_train = np.full((X_train.shape[0],), component_labels_dict[component_label])
    y_train = to_categorical(y_train, num_classes=len(component_labels))

    return X_train, y_train

def detect_anomalies(data_combined, timestamps_combined, temp_threshold, board_voltage_threshold, voltage_threshold):
    temp_values = data_combined[:, 0] 
    board_voltage_values = data_combined[:, 1]
    voltage_values = data_combined[:, 2]
    
    temp_highs = temp_values > temp_threshold
    board_voltage_lows = board_voltage_values < board_voltage_threshold
    voltage_lows = voltage_values < voltage_threshold
    
    temp_anomaly_timestamps = np.array(timestamps_combined)[temp_highs]
    board_voltage_anomaly_timestamps = np.array(timestamps_combined)[board_voltage_lows]
    voltage_anomaly_timestamps = np.array(timestamps_combined)[voltage_lows]

    window_size = 60
    tolerance = 5
    sustained_anomalies = np.zeros_like(temp_highs, dtype=bool)

    for i in range(len(temp_highs) - window_size + 1):
        window_values = data_combined[i:i + window_size, 0].flatten() 
        if np.sum(temp_highs[i:i + window_size] & (window_values > temp_threshold)) >= (window_size - tolerance):
            sustained_anomalies[i:i + window_size] = (window_values > temp_threshold)

    sustained_anomaly_timestamps = np.array(timestamps_combined)[sustained_anomalies]

    return {
        'temp_values': temp_values,
        'board_voltage_values': board_voltage_values,
        'voltage_values': voltage_values,
        'temp_anomaly_timestamps': temp_anomaly_timestamps,
        'board_voltage_anomaly_timestamps': board_voltage_anomaly_timestamps,
        'voltage_anomaly_timestamps': voltage_anomaly_timestamps,
        'sustained_anomaly_timestamps': sustained_anomaly_timestamps,
        'temp_highs': temp_highs,
        'board_voltage_lows': board_voltage_lows,
        'voltage_lows': voltage_lows,
        'sustained_anomalies': sustained_anomalies,
    }

def predict_faulty_component(data_combined, component_labels, model, temp_threshold, board_voltage_threshold, voltage_threshold):
    last_hour_data = data_combined[-60:]
    faulty_component = "None"
    accuracy = 0.0
    alert1 = alert2 = alert3 = alert4 = False

    temp_values_last_hour = last_hour_data[:, 0] 
    board_voltage_values_last_hour = last_hour_data[:, 1]
    voltage_values_last_hour = last_hour_data[:, 2]
    temp_highs_last_hour = temp_values_last_hour > temp_threshold
    board_voltage_lows_last_hour = board_voltage_values_last_hour < board_voltage_threshold
    voltage_lows_last_hour = voltage_values_last_hour < voltage_threshold

    if len(last_hour_data) >= 0:
        X_sustained_last_hour = last_hour_data.reshape(-1, 1, 3)
        diffs_last_hour = X_sustained_last_hour - [temp_threshold, board_voltage_threshold, voltage_threshold]
        X_sustained_diff_last_hour = diffs_last_hour.reshape(-1, 1, 3)

        predictions = model.predict(X_sustained_diff_last_hour)
        num_classes = predictions.shape[1]
        
        if num_classes > 0:
            faulty_component_idx = np.argmax(predictions, axis=1) 
            most_common_idx = np.bincount(faulty_component_idx).argmax()
            if 0 <= most_common_idx < len(component_labels):
                faulty_component = component_labels[most_common_idx]
                accuracy = round(np.mean(np.max(predictions, axis=1)) * 100, 2)
                if accuracy < 65:
                    faulty_component = "Unknown"
                    accuracy = 0.0

        # Conditions for setting alerts
        if np.all(temp_highs_last_hour) == True and (temp_highs_last_hour).size == 35:
            alert1 = True
        elif np.all(board_voltage_lows_last_hour) == True and (board_voltage_lows_last_hour).size == 45:
            alert2 = True
        elif np.all(temp_highs_last_hour) == True and (temp_highs_last_hour).size == 50:
            alert3 = True
        elif np.all(voltage_lows_last_hour) == True and (voltage_lows_last_hour).size == 59:
            alert4 = True
        else:
            alert1 = False
            alert2 = False
            alert3 = False
            alert4 = False

    return faulty_component, accuracy, alert1, alert2, alert3, alert4
