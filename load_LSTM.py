import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def normalize_data(data_path):
    # Load the dataset
    df = pd.read_csv(data_path)

    # Convert the 'Timestamp' column to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Convert timestamps to the desired format
    df['Formatted_Timestamp'] = df['Timestamp'].dt.strftime('%m/%d/%Y %H:%M:%S')

    # Sort the DataFrame by 'Timestamp'
    df = df.sort_values(by='Timestamp').reset_index(drop=True)

    # Drop the specified columns
    data = df.drop(columns=['Timestamp', 'Formatted_Timestamp', 'File', 'Acceleration_X', 'Acceleration_Y', 'Acceleration_Z', 'Altitude', 
                            'Ambient_Temp', 'GPS_Fix', 'Humidity', 'Infrared', 'Latitude', 'Light', 'Longitude', 
                            'Magnetometer_X', 'Magnetometer_Y', 'Magnetometer_Z', 'Visible'])

    # Extract the formatted timestamp column
    timestamps = df['Formatted_Timestamp']

    # Print data statistics before normalization
    print("Data statistics before normalization:")
    print(data.describe())

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    # Print data statistics after normalization
    print("Data statistics after normalization:")
    print(pd.DataFrame(data).describe())

    return data, scaler, timestamps

# Function to create dataset
def create_dataset(data, timestamps, time_step=1):
    X, y, ts = [], [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :])
        y.append(data[i + time_step, :])
        ts.append(timestamps.iloc[i + time_step])
    return np.array(X), np.array(y), np.array(ts)

def main():
    # Define time step
    time_step = 10

    data, scaler, timestamps = normalize_data('output_fake.csv')

    # Create dataset
    X, y, ts = create_dataset(data, timestamps, time_step)
    print(X.shape, y.shape, ts.shape)

    model = tf.keras.models.load_model('LSTM.keras')

    # Make predictions on the entire dataset
    predictions = model.predict(X)

    # Inverse transform the predictions and actual values
    predictions_inv = scaler.inverse_transform(predictions)
    y_inv = scaler.inverse_transform(y)

    # Calculate the reconstruction error
    error = np.mean(np.abs(predictions_inv - y_inv), axis=1)
    temp_values = y_inv[:, 0]

    # Determine an error threshold for anomalies
    threshold = np.mean(error) + 5 * np.std(error)

    with open('threshold.txt', 'r') as f:
        temp_threshold = float(f.read())

    # Detect anomalies based on reconstruction error
    error_anomalies = error > threshold

    # Detect anomalies based on actual Probe_Temp values
    temp_highs = temp_values > temp_threshold  # Assuming Probe_Temp is the first column

    # Print the number of anomalies detected
    print(f'Number of anomalies detected based on reconstruction error: {np.sum(error_anomalies)}')
    print(f'Number of anomalies detected based on Probe_Temp: {np.sum(temp_highs)}')

    # Combine anomalies with their corresponding timestamps
    error_anomaly_timestamps = ts[error_anomalies]
    temp_anomaly_timestamps = ts[temp_highs]

    # Print anomaly timestamps
    print("Anomalies detected at timestamps based on Probe_Temp:")
    print(temp_anomaly_timestamps)

    # Post-process to check for sustained anomalies
    sustained_anomalies = np.zeros(len(temp_highs), dtype=bool)
    window_size = 60  # 60 minutes for one hour

    for i in range(len(temp_highs) - window_size + 1):
        if np.all(temp_highs[i:i + window_size]):
            sustained_anomalies[i:i + window_size] = True

    # Get timestamps for sustained anomalies
    sustained_anomaly_timestamps = ts[sustained_anomalies]

    # Print sustained anomaly timestamps
    print("Sustained anomalies detected at timestamps:")
    print(sustained_anomaly_timestamps)

    # Plotting the results
    plt.figure(figsize=(15, 5))
    plt.plot(ts, y_inv[:, 0], label='Actual')
    plt.axhline(y=temp_threshold, color='r', linestyle='--', label='Threshold')
    plt.scatter(temp_anomaly_timestamps, y_inv[temp_highs, 0], color='orange', label=f'Temp Highs (Probe_Temp > {temp_threshold:.2f})')
    plt.scatter(sustained_anomaly_timestamps, y_inv[sustained_anomalies, 0], color='red', label='Sustained Anomalies')
    
    plt.xlabel('Timestamp')
    plt.ylabel('Probe_Temp')
    plt.xticks(rotation=45)
    
    # Reduce the number of x-axis labels
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
    
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
