from tensorflow import keras
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout
from keras._tf_keras.keras.optimizers import Adam
import os

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(1, 1)))  # Adjust input shape as needed
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(4, activation='softmax'))  # Output layer for 4 classes with softmax activation

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.save('LSTM.keras')

component_file = 'components.txt'
initial_components = ['compressor', 'evaporator', 'condenser', 'thermocouple']

# Create the file with initial components if it doesn't exist
if not os.path.exists(component_file):
    with open(component_file, 'w') as f:
        for component in initial_components:
            f.write(f"{component}\n")