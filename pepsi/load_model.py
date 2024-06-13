import tensorflow as tf
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler

def load_data(data_dir):
    features = []
    labels = []

    ACCEPTABLE_MIN = 4
    ACCEPTABLE_MAX = 20

    for file_name in os.listdir(data_dir):
        if file_name.endswith('.json'):
            with open(os.path.join(data_dir, file_name), 'r') as f:
                data = json.load(f) 
                sensor_values = data.get('sensorValues', {})
                if 'Probe_Temp' in sensor_values and sensor_values['Probe_Temp']:
                    probe_temp_value = float(sensor_values['Probe_Temp'][0]['value'])
                    feature = [float(sensor_values[key][0]['value']) for key in sensor_values if key not in ['Probe_Temp', 'Motion'] and sensor_values[key]]
                    label = 1 if ACCEPTABLE_MIN <= probe_temp_value <= ACCEPTABLE_MAX else 0
                    features.append(feature)
                    labels.append([label])

    return np.array(features), np.array(labels)

def main():
    data_dir = r'records'

    X, y = load_data(data_dir)

    print(f'Loaded data shape: X={X.shape}, y={y.shape}')

    # Load the model
    model = tf.keras.models.load_model('sensor_model_with_classification.keras')

    # Normalize the features
    scaler_X = StandardScaler()
    X_normalized = scaler_X.fit_transform(X)

    print(f'Normalized data shape: X_normalized={X_normalized.shape}')

    # Predict with the model
    predictions = model.predict(X_normalized)

    # Since we are predicting binary classification, we do not need to inverse transform
    predictions = (predictions > 0.5).astype(int)

    for i in range(200):
        print(f'Actual: {y[i]}, Predicted: {predictions[i]}')

if __name__ == '__main__':
    main()



