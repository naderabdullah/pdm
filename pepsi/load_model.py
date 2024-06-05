import tensorflow as tf
import numpy as np
import os
import json
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

    X, y = load_data(data_dir, 'Probe_Temp')

    # load the model
    model = tf.keras.models.load_model('sensor_model.keras')

    scaler_X = StandardScaler()
    X_normalized = scaler_X.fit_transform(X)

    # predict with the model
    predictions = model.predict(X_normalized)

    scaler_y = StandardScaler()
    y_normalized = scaler_y.fit_transform(y)
    predictions_inversed = np.round(scaler_y.inverse_transform(predictions), 2)

    for i in range(10):
        print(f'Prediction: {predictions_inversed[i]}')

if __name__ == '__main__':
    main()
