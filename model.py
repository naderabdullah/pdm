from tensorflow import keras
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout, SimpleRNN
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.regularizers import l2
import pandas as pd
import os

def update_combined_data():
    global df_combined
    try:
        df_new = pd.read_csv('new_data.csv')
        if not df_new.empty:
            new_record = df_new.iloc[0:1]
            # Append the new record to output.csv
            new_record.to_csv('output.csv', mode='a', header=False, index=False)
            # Reload df_combined from output.csv
            df_combined = pd.read_csv('output.csv')
            # Remove the appended record from new_data.csv
            df_new = df_new.iloc[1:]
            df_new.to_csv('new_data.csv', index=False)
    except pd.errors.EmptyDataError:
        df_new = pd.DataFrame()

model = Sequential()
model.add(SimpleRNN(50, return_sequences=True, input_shape=(1, 1), kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(SimpleRNN(50, return_sequences=False, kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax', kernel_regularizer=l2(0.01)))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.save('LSTM.keras')

component_file = 'components.txt'
initial_components = ['compressor', 'evaporator', 'condenser', 'thermocouple']

# Create the file with initial components if it doesn't exist
if not os.path.exists(component_file):
    with open(component_file, 'w') as f:
        for component in initial_components:
            f.write(f"{component}\n")