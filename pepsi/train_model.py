import json
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

# Define acceptable range
ACCEPTABLE_MIN = 4
ACCEPTABLE_MAX = 20

def load_data(data_dir):
    features = []
    range_labels = []

    for file_name in os.listdir(data_dir):
        if file_name.endswith('.json'):
            with open(os.path.join(data_dir, file_name), 'r') as f:
                data = json.load(f)
                sensor_values = data.get('sensorValues', {})
                feature = [float(sensor_values[key][0]['value']) for key in sensor_values if key not in ['Probe_Temp', 'Motion'] and sensor_values[key]]
                if 'Probe_Temp' in sensor_values and sensor_values['Probe_Temp']:
                    probe_temp = float(sensor_values['Probe_Temp'][0]['value'])
                    range_label = 1 if ACCEPTABLE_MIN <= probe_temp <= ACCEPTABLE_MAX else 0
                    features.append(feature)
                    range_labels.append(range_label)

    return np.array(features), np.array(range_labels)

def main():
    data_dir = r'records'

    # Load the data
    X, y = load_data(data_dir)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalization
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    # Calculate class weights to handle class imbalance
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))

    # Define the model
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='nadam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=500, validation_split=0.2, batch_size=32, class_weight=class_weights_dict)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    
    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')

    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    for i in range(10):
        print(f'Actual: {y_test[i]}, Predicted: {y_pred[i]}')

    model.save('sensor_model_with_classification.keras')

if __name__ == '__main__':
    main()






