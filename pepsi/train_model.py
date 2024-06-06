import json
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(data_dir, target):
    features = []
    labels = []

    for file_name in os.listdir(data_dir):
        if file_name.endswith('.json'):
            with open(os.path.join(data_dir, file_name), 'r') as f:
                data = json.load(f)
                feature = [float(data[key]) for key in data if key != 'File' and key != 'Motion']
                label = [float(data['Ambient_Temp']), float(data['Probe_Temp'])]
                features.append(feature)
                labels.append(label)

    return np.array(features), np.array(labels)

def main():
    data_dir = r'generated_records'

    #load the data
    X, y = load_data(data_dir, 'Probe_Temp')

    #split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #normalization
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.fit_transform(X_test)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.fit_transform(y_test)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='softmax', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='softmax'),
        tf.keras.layers.Dense(y_train.shape[1])
    ])

    model.compile(optimizer='nadam', loss='mean_squared_error', metrics=['mae'])

    history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=32)

    loss, mae = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')
    print(f'Test MAE: {mae}')

    y_test_inversed = scaler_y.inverse_transform(y_test)
    for i in range(10):
        print(f'Actual: {y_test_inversed[i]}')

    model.save('sensor_model.keras')

if __name__ == '__main__':
    main()

# Load the model for future use
# model = tf.keras.models.load_model('sensor_model.keras')

# Predict with the model
# predictions = model.predict(X_test)
