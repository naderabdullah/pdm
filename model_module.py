from tensorflow import keras
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.callbacks import TensorBoard
from datetime import datetime

log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

def load_trained_model():
    return load_model('LSTM.keras')

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=1, callbacks=[tensorboard_callback])
