import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('output.csv')

# Convert the 'Timestamp' column to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Sort the DataFrame by 'Timestamp'
df = df.sort_values(by='Timestamp').reset_index(drop=True)

# Drop the specified columns
data = df.drop(columns=['Timestamp', 'File', 'Acceleration_X', 'Acceleration_Y', 'Acceleration_Z', 'Altitude', 
                        'Ambient_Temp', 'GPS_Fix', 'Humidity', 'Infrared', 'Latitude', 'Light', 'Longitude', 
                        'Magnetometer_X', 'Magnetometer_Y', 'Magnetometer_Z', 'Visible'])

# Extract the timestamp column
timestamps = df['Timestamp']

# Print data statistics before normalization
print("Data statistics before normalization:")
print(data.describe())

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# Print data statistics after normalization
print("Data statistics after normalization:")
print(pd.DataFrame(data).describe())

# Function to create dataset
def create_dataset(data, time_step=1):
    X, y, ts = [], [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :])
        y.append(data[i + time_step, :])
        ts.append(timestamps.iloc[i + time_step])
    return np.array(X), np.array(y), np.array(ts)

# Define time step
time_step = 10

# Create dataset
X, y, ts = create_dataset(data, time_step)
print(X.shape, y.shape, ts.shape)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(1, 1)))  # Adjust input shape as needed
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))  # Output layer for 3 classes with softmax activation

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.save('LSTM.keras')

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
temp_threshold = np.percentile(temp_values, 99)

f = open('threshold.txt', 'w')
f.write(f'{temp_threshold:.2f}')
f.close()

# Detect anomalies based on reconstruction error
error_anomalies = error > threshold

# Detect anomalies based on actual Probe_Temp values
temp_anomalies = temp_values > temp_threshold  # Assuming Probe_Temp is the first column

# Print the number of anomalies detected
print(f'Number of anomalies detected based on reconstruction error: {np.sum(error_anomalies)}')
print(f'Number of anomalies detected based on Probe_Temp: {np.sum(temp_anomalies)}')

# Combine anomalies with their corresponding timestamps
error_anomaly_timestamps = ts[error_anomalies]
temp_anomaly_timestamps = ts[temp_anomalies]

# Print anomaly timestamps
#print("Anomalies detected at timestamps based on reconstruction error:")
#print(error_anomaly_timestamps)
print("Anomalies detected at timestamps based on Probe_Temp:")
print(temp_anomaly_timestamps)

# Plotting the results
plt.figure(figsize=(15, 5))
plt.plot(ts, y_inv[:, 0], label='Actual')
plt.axhline(y=temp_threshold, color='r', linestyle='--', label='Threshold')
plt.scatter(temp_anomaly_timestamps, y_inv[temp_anomalies, 0], color='red', label=f'Anomalies (Probe_Temp > {temp_threshold:.2f})')
plt.xlabel('Timestamp')
plt.ylabel('Probe_Temp')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Plotting the reconstruction error and anomalies
plt.figure(figsize=(15, 5))
plt.plot(ts, error, label='Reconstruction error')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.scatter(ts[error_anomalies], error[error_anomalies], color='red', label='Anomalies (Reconstruction error)')
plt.xlabel('Timestamp')
plt.ylabel('Reconstruction Error')
plt.xticks(rotation=45)
plt.legend()
plt.show()